import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class ShapeEvalResult:
    shape_idx: int
    orig_label: str
    aug_label: str
    n_points_orig: int
    n_points_aug: int
    label_match: bool
    recovered_vertices: int
    cap_strict: float
    cap_bidir: float
    align_rmse: float
    mean_nn_dist: float
    distance_threshold: float
    recovered_indices: List[int]


@dataclass
class DatasetEvalResult:
    matched_file_pairs: int
    total_instances: int
    dataset_mean_cap: Optional[float]
    dataset_mean_cap_bidir: Optional[float]


class VisualizationHelper:
    """Shared visualization class."""

    @staticmethod
    def close_poly(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        if np.allclose(pts[0], pts[-1]):
            return pts
        return np.vstack([pts, pts[0]])

    def visualize_one_polygon(
        self,
        orig_pts: np.ndarray,
        aug_pts: np.ndarray,
        recovered_idx: List[int],
        aug_to_orig_idx: List[Optional[int]],
        orig_aligned_pts: np.ndarray,
        out_path: Path,
        title: str = "",
    ) -> None:
        n = len(orig_pts)
        m = len(recovered_idx)

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(*self.close_poly(orig_pts).T, linewidth=1.1, color="#4C78A8", alpha=0.45, label="Original")
        ax1.plot(*self.close_poly(orig_aligned_pts).T, linewidth=1.0, linestyle="--", color="#2F4B7C", alpha=0.9, label="Original aligned")
        ax1.plot(*self.close_poly(aug_pts).T, linewidth=1.2, color="#F58518", alpha=0.95, label="Augmented")
        if len(orig_aligned_pts) > 0:
            ax1.scatter(orig_aligned_pts[:, 0], orig_aligned_pts[:, 1], s=10, color="#2F4B7C", alpha=0.9)
        if len(aug_pts) > 0:
            ax1.scatter(aug_pts[:, 0], aug_pts[:, 1], s=12, color="#E45756", alpha=0.95)

        # NN mapping lines: augmented vertex -> aligned original vertex.
        for p, idx in zip(aug_pts, aug_to_orig_idx):
            if idx is None:
                continue
            q = orig_aligned_pts[idx]
            ax1.plot([p[0], q[0]], [p[1], q[1]], linewidth=0.5, color="#6B6B6B", alpha=0.8)

        # Annotate original vertex ids.
        for oi, p in enumerate(orig_aligned_pts):
            ax1.annotate(
                f"o{oi}",
                (p[0], p[1]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=10,
                color="#1F355A",
            )

        # Annotate augmented vertex ids. For dense polygons, annotate matched points only.
        dense_aug = len(aug_pts) > max(3 * max(1, len(orig_pts)), 60)
        for ai, p in enumerate(aug_pts):
            oi = aug_to_orig_idx[ai] if ai < len(aug_to_orig_idx) else None
            if dense_aug and oi is None:
                continue
            text = f"a{ai}" if oi is None else f"a{ai}->o{oi}"
            ax1.annotate(
                text,
                (p[0], p[1]),
                textcoords="offset points",
                xytext=(3, -8),
                fontsize=10,
                color="#7A2D00",
            )

        ax1.set_aspect("equal", adjustable="box")
        ax1.legend()
        ax1.grid(False)

        ax2 = fig.add_subplot(1, 2, 2)
        if m == 0:
            ax2.text(0.5, 0.5, "No recovered indices", ha="center", va="center", fontsize=10)
            ax2.axis("off")
        else:
            angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
            xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)

            for t in range(m):
                ax2.scatter([xy[t, 0]], [xy[t, 1]], s=55, color="#4C78A8")
                ax2.text(xy[t, 0], xy[t, 1], str(recovered_idx[t]), ha="center", va="center", fontsize=8, color="white")

            preserved = 0
            for t in range(m):
                a = recovered_idx[t]
                b = recovered_idx[(t + 1) % m]
                ok = b == (a + 1) % n
                preserved += int(ok)
                x0, y0 = xy[t]
                x1, y1 = xy[(t + 1) % m]
                ax2.plot([x0, x1], [y0, y1], linewidth=2.0, color="#54A24B" if ok else "#E45756", linestyle="-" if ok else "--")

            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(False)
            ax2.axis("off")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


class EvaluationDataSaver:
    """Shared save class for text/csv patterns."""

    @staticmethod
    def sanitize_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)

    @staticmethod
    def save_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    @staticmethod
    def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


class BaseCapProcessor:
    """Base CAP processor: shared CAP math + __call__ contract."""

    def __init__(self, distance_threshold: float, logger: logging.Logger, visualizer: VisualizationHelper, saver: EvaluationDataSaver):
        self.distance_threshold = distance_threshold
        self.logger = logger
        self.visualizer = visualizer
        self.saver = saver

    def __call__(self, args: argparse.Namespace) -> Optional[DatasetEvalResult]:
        raise NotImplementedError("Subclasses must implement __call__")

    @staticmethod
    def compute_cap_strict(indices: List[int], n_original: int) -> float:
        m = len(indices)
        if m < 2:
            return 0.0
        ok = 0
        for t in range(m):
            a = indices[t]
            b = indices[(t + 1) % m]
            if b == (a + 1) % n_original:
                ok += 1
        return ok / m

    @staticmethod
    def compute_cap_bidirectional(indices: List[int], n_original: int) -> float:
        m = len(indices)
        if m < 2:
            return 0.0
        ok = 0
        for t in range(m):
            a = indices[t]
            b = indices[(t + 1) % m]
            if b == (a + 1) % n_original or b == (a - 1) % n_original:
                ok += 1
        return ok / m

    def save_paper_plots(self, records: List[Dict[str, object]], out_dir: Path, mode_name: str) -> None:
        if not records:
            return
        plots_dir = out_dir / "paper_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        cap_s = np.array([float(r["cap_strict"]) for r in records], dtype=np.float64)
        cap_b = np.array([float(r["cap_bidir"]) for r in records], dtype=np.float64)
        npts = np.array([float(r["n_points_orig"]) for r in records], dtype=np.float64)
        rec = np.array([float(r["recovered_vertices"]) for r in records], dtype=np.float64)
        rmse = np.array([float(r["align_rmse"]) for r in records], dtype=np.float64)
        mnn = np.array([float(r["mean_nn_dist"]) for r in records], dtype=np.float64)
        labels = [str(r["label"]) for r in records]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        vp = ax.violinplot([cap_s, cap_b], showmeans=True, showmedians=True, widths=0.8)
        for b in vp["bodies"]:
            b.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["CAP strict", "CAP bidirectional"], fontsize=12)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{mode_name.upper()} CAP Distributions", fontsize=14)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / "cap_distributions.png", dpi=220)
        plt.close(fig)

        ratio = np.divide(rec, np.maximum(npts, 1.0))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(npts, ratio, s=18, alpha=0.45, c=cap_s, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Original polygon vertices", fontsize=13)
        ax.set_ylabel("Recovered ratio (recovered / original)", fontsize=13)
        ax.set_title(f"{mode_name.upper()} Recovery vs Polygon Complexity", fontsize=14)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / "recovery_vs_complexity.png", dpi=220)
        plt.close(fig)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        rmse_f = rmse[np.isfinite(rmse)]
        if len(rmse_f) > 0:
            ax1.hist(rmse_f, bins=28, color="#4C78A8", alpha=0.85)
        ax1.set_title("Alignment RMSE distribution", fontsize=13)
        ax1.set_xlabel("RMSE (pixels)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.grid(alpha=0.2)

        ax2 = fig.add_subplot(1, 2, 2)
        mnn_f = mnn[np.isfinite(mnn)]
        if len(mnn_f) > 0:
            ax2.hist(mnn_f, bins=28, color="#F58518", alpha=0.85)
        ax2.set_title("Mean NN distance distribution", fontsize=13)
        ax2.set_xlabel("Distance (pixels)", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / "alignment_quality_histograms.png", dpi=220)
        plt.close(fig)

        uniq = sorted(set(labels))
        mat = np.zeros((2, len(uniq)), dtype=np.float64)
        cnt = np.zeros(len(uniq), dtype=np.int32)
        idx = {k: i for i, k in enumerate(uniq)}
        for r in records:
            i = idx[str(r["label"])]
            mat[0, i] += float(r["cap_strict"])
            mat[1, i] += float(r["cap_bidir"])
            cnt[i] += 1
        for i in range(len(uniq)):
            if cnt[i] > 0:
                mat[:, i] /= float(cnt[i])
        fig = plt.figure(figsize=(max(10, len(uniq) * 0.9), 4.8))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["CAP strict", "CAP bidirectional"], fontsize=12)
        ax.set_xticks(np.arange(len(uniq)))
        ax.set_xticklabels(uniq, rotation=35, ha="right", fontsize=11)
        ax.set_title(f"{mode_name.upper()} Class-wise CAP", fontsize=14)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r, c]:.2f}\n(n={cnt[c]})", ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "classwise_cap_heatmap.png", dpi=220)
        plt.close(fig)

        failures = sorted(records, key=lambda x: float(x["cap_strict"]))[:9]
        valid_failures = []
        for r in failures:
            pstr = str(r.get("visualization_path", "")).strip()
            if not pstr:
                continue
            p = Path(pstr)
            if p.exists() and p.is_file():
                valid_failures.append(r)
        if valid_failures:
            cols = 3
            rows = int(np.ceil(len(valid_failures) / cols))
            fig = plt.figure(figsize=(cols * 5.2, rows * 4.2))
            for i, r in enumerate(valid_failures):
                ax = fig.add_subplot(rows, cols, i + 1)
                img = plt.imread(str(r["visualization_path"]))
                ax.imshow(img)
                ax.set_title(
                    f"{r['file_pair']}\n{r['label']} | CAP={float(r['cap_strict']):.3f}",
                    fontsize=11,
                )
                ax.axis("off")
            fig.suptitle(f"{mode_name.upper()} Failure Gallery (Lowest CAP strict)", fontsize=14)
            fig.tight_layout()
            fig.savefig(plots_dir / "failure_gallery.png", dpi=220)
            plt.close(fig)


class RoboFlowProcessor(BaseCapProcessor):
    """Roboflow evaluator using geometric matching."""

    def load_labelme_polygons(self, path: Path) -> List[Dict]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [s for s in data.get("shapes", []) if s.get("shape_type") == "polygon"]

    def find_matching_files(self, orig_file: Path, roboflow_dir: Path) -> List[Path]:
        base = orig_file.stem
        return sorted([p for p in roboflow_dir.glob("*.json") if p.stem.startswith(base + "_png")])

    @staticmethod
    def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst
        cov = (dst_c.T @ src_c) / src.shape[0]
        u, svals, vt = np.linalg.svd(cov)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        var_src = (src_c ** 2).sum() / src.shape[0]
        scale = (svals.sum() / var_src) if var_src > 1e-12 else 1.0
        t = mu_dst - scale * (r @ mu_src)
        return r, scale, t

    @staticmethod
    def resample_closed_polygon(points: np.ndarray, m: int) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return np.zeros((m, 2), dtype=np.float64)
        if len(pts) == 1:
            return np.repeat(pts, m, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        seg = np.linalg.norm(nxt - pts, axis=1)
        total = float(seg.sum())
        if total <= 1e-12:
            return np.repeat(pts[:1], m, axis=0)
        csum = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(0.0, total, m, endpoint=False)
        out = np.zeros((m, 2), dtype=np.float64)
        for i, t in enumerate(targets):
            k = int(np.searchsorted(csum, t, side="right") - 1)
            k = min(k, len(pts) - 1)
            d = seg[k]
            a = 0.0 if d <= 1e-12 else (t - csum[k]) / d
            out[i] = pts[k] + a * (nxt[k] - pts[k])
        return out

    def best_similarity_with_cyclic_options(self, original_pts: np.ndarray, augmented_pts: np.ndarray, max_samples: int = 64):
        m = min(len(original_pts), len(augmented_pts), max_samples)
        src = self.resample_closed_polygon(original_pts, m)
        dst = self.resample_closed_polygon(augmented_pts, m)
        best_rmse = np.inf
        best_r = np.eye(2, dtype=np.float64)
        best_s = 1.0
        best_t = np.zeros(2, dtype=np.float64)
        for reverse in (False, True):
            d = dst[::-1] if reverse else dst
            for shift in range(m):
                ds = np.roll(d, -shift, axis=0)
                r, s, t = self.umeyama_similarity(src, ds)
                st = (s * (src @ r.T)) + t
                rmse = float(np.sqrt(np.mean(np.sum((st - ds) ** 2, axis=1))))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_r, best_s, best_t = r, s, t
        return best_r, best_s, best_t, best_rmse

    @staticmethod
    def auto_distance_threshold(augmented_pts: np.ndarray) -> float:
        span = np.ptp(augmented_pts, axis=0)
        return max(2.0, 0.035 * float(np.linalg.norm(span)))

    def recover_indices_aligned_nn(self, original_pts: np.ndarray, augmented_pts: np.ndarray, distance_threshold: float = 0.0):
        r, s, t, align_rmse = self.best_similarity_with_cyclic_options(original_pts, augmented_pts)
        orig_aligned = (s * (original_pts @ r.T)) + t
        dmat = np.linalg.norm(augmented_pts[:, None, :] - orig_aligned[None, :, :], axis=2)

        def solve_with_threshold(thresh: float):
            cost = dmat.copy()
            cost[cost > thresh] = 1e6
            cost = cost + (np.arange(cost.shape[1], dtype=np.float64) * 1e-9)[None, :]
            rows, cols = linear_sum_assignment(cost)
            aug_to_orig_idx: List[Optional[int]] = [None] * len(augmented_pts)
            aug_to_orig_dist: List[float] = [float("nan")] * len(augmented_pts)
            for ai, oi in zip(rows, cols):
                dist = float(dmat[ai, oi])
                if dist <= thresh:
                    aug_to_orig_idx[int(ai)] = int(oi)
                    aug_to_orig_dist[int(ai)] = dist
            rec_idx: List[int] = []
            rec_dist: List[float] = []
            for i, d in zip(aug_to_orig_idx, aug_to_orig_dist):
                if i is None:
                    continue
                rec_idx.append(i)
                rec_dist.append(d)
            return rec_idx, rec_dist, aug_to_orig_idx

        if distance_threshold > 0:
            used = distance_threshold
            rec_idx, rec_dist, aug_to_orig_idx = solve_with_threshold(used)
            return rec_idx, rec_dist, aug_to_orig_idx, orig_aligned, align_rmse, used

        base = self.auto_distance_threshold(augmented_pts)
        candidate_factors = [1.0, 1.2, 1.4, 1.7, 2.0]
        candidates = []
        for f in candidate_factors:
            th = base * f
            rec_idx, rec_dist, aug_to_orig_idx = solve_with_threshold(th)
            mean_d = float(np.mean(rec_dist)) if rec_dist else float("inf")
            candidates.append((th, rec_idx, rec_dist, aug_to_orig_idx, mean_d))

        max_rec = max(len(c[1]) for c in candidates) if candidates else 0
        min_acceptable = max(1, max_rec - 1)
        acceptable = [c for c in candidates if len(c[1]) >= min_acceptable]
        if not acceptable:
            acceptable = candidates
        chosen = min(acceptable, key=lambda c: (c[0], c[4]))
        used, rec_idx, rec_dist, aug_to_orig_idx, _ = chosen
        return rec_idx, rec_dist, aug_to_orig_idx, orig_aligned, align_rmse, used

    def build_shape_summary_text(self, orig_file: Path, aug_file: Path, orig_shapes: List[Dict], aug_shapes: List[Dict]) -> str:
        lines = [
            "================ SHAPE SUMMARY ================",
            f"Original:  {orig_file.name} | polygon shapes = {len(orig_shapes)}",
            f"Roboflow:  {aug_file.name} | polygon shapes = {len(aug_shapes)}",
            "------------------------------------------------",
            "idx | label | n_points_orig | n_points_roboflow",
            "------------------------------------------------",
        ]
        m = min(len(orig_shapes), len(aug_shapes))
        for i in range(m):
            lines.append(
                f"{i:>3} | {orig_shapes[i].get('label','')} / {aug_shapes[i].get('label','')} | "
                f"{len(orig_shapes[i]['points']):>12} | {len(aug_shapes[i]['points']):>16}"
            )
        lines.append("================================================")
        return "\n".join(lines)

    def evaluate_shape(self, shape_idx: int, orig_shape: Dict, aug_shape: Dict) -> Tuple[ShapeEvalResult, Dict[str, object]]:
        o_pts = np.array(orig_shape["points"], dtype=np.float64)
        a_pts = np.array(aug_shape["points"], dtype=np.float64)
        rec_idx, rec_dist, aug_to_orig_idx, orig_aligned, rmse, used = self.recover_indices_aligned_nn(
            o_pts, a_pts, self.distance_threshold
        )
        cap = self.compute_cap_strict(rec_idx, len(o_pts))
        cap_b = self.compute_cap_bidirectional(rec_idx, len(o_pts))
        mean_dist = float(np.mean(rec_dist)) if rec_dist else float("nan")
        result = ShapeEvalResult(
            shape_idx=shape_idx,
            orig_label=str(orig_shape.get("label", "")),
            aug_label=str(aug_shape.get("label", "")),
            n_points_orig=len(o_pts),
            n_points_aug=len(a_pts),
            label_match=str(orig_shape.get("label", "")) == str(aug_shape.get("label", "")),
            recovered_vertices=len(rec_idx),
            cap_strict=cap,
            cap_bidir=cap_b,
            align_rmse=rmse,
            mean_nn_dist=mean_dist,
            distance_threshold=used,
            recovered_indices=rec_idx,
        )
        return result, {"o_pts": o_pts, "a_pts": a_pts, "aug_to_orig_idx": aug_to_orig_idx, "orig_aligned": orig_aligned}

    def evaluate_single(self, orig_dir: Path, roboflow_dir: Path, file_name: str, shape_idx: int, out_path: Path, roboflow_file: str = "") -> None:
        orig_file = orig_dir / file_name
        if not orig_file.exists():
            raise FileNotFoundError(f"Original JSON not found: {orig_file}")
        if roboflow_file:
            aug_file = roboflow_dir / roboflow_file
            if not aug_file.exists():
                raise FileNotFoundError(f"Specified Roboflow JSON not found: {aug_file}")
        else:
            matches = self.find_matching_files(orig_file, roboflow_dir)
            if not matches:
                raise FileNotFoundError(f"No Roboflow match found for: {orig_file.name}")
            aug_file = matches[0]
        orig_shapes = self.load_labelme_polygons(orig_file)
        aug_shapes = self.load_labelme_polygons(aug_file)
        print("\n" + self.build_shape_summary_text(orig_file, aug_file, orig_shapes, aug_shapes))
        print("======== CURRENT JSON EVALUATION ========")
        m = min(len(orig_shapes), len(aug_shapes))
        if shape_idx < 0:
            print("Set --shape-idx to visualize a specific polygon.")
            return
        if shape_idx >= m:
            raise IndexError(f"shape-idx {shape_idx} out of range (max {m - 1})")
        result, dbg = self.evaluate_shape(shape_idx, orig_shapes[shape_idx], aug_shapes[shape_idx])
        title = (
            f"{orig_file.name} | Roboflow: {aug_file.name}\n"
            f"shape_idx={result.shape_idx} | recovered m={result.recovered_vertices} | CAP={result.cap_strict:.4f}\n"
            f"align_rmse={result.align_rmse:.2f} | mean_nn_dist={result.mean_nn_dist:.2f} | thresh={result.distance_threshold:.2f}"
        )
        self.visualizer.visualize_one_polygon(
            dbg["o_pts"], dbg["a_pts"], result.recovered_indices, dbg["aug_to_orig_idx"], dbg["orig_aligned"], out_path, title
        )
        print(f"Saved: {out_path}")
        print(f"Recovered index sequence: {result.recovered_indices}")
        print(f"CAP (strict): {result.cap_strict:.4f}")
        print(f"CAP (bidirectional): {result.cap_bidir:.4f}")
        print(f"Recovered vertices: {result.recovered_vertices} / {result.n_points_orig}")
        print(f"Alignment RMSE: {result.align_rmse:.4f}")
        print(f"Mean NN distance: {result.mean_nn_dist:.4f}")
        print(f"Distance threshold used: {result.distance_threshold:.4f}")
        print("========================================")

    def evaluate_all_pairs(self, orig_dir: Path, roboflow_dir: Path, out_dir: Path) -> DatasetEvalResult:
        orig_files = sorted(orig_dir.glob("*.json"))
        if not orig_files:
            raise FileNotFoundError(f"No JSON files in original dir: {orig_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        total_instances = 0
        total_cap = 0.0
        total_cap_bidir = 0.0
        file_pairs = 0
        plot_records: List[Dict[str, object]] = []
        final_lines: List[str] = [f"Distance threshold: {self.distance_threshold}", "========================================", "======== DATASET CAP EVALUATION ========"]
        print(final_lines[0]); print(final_lines[1]); print(final_lines[2])
        for orig_file in orig_files:
            matches = self.find_matching_files(orig_file, roboflow_dir)
            if not matches:
                continue
            orig_shapes = self.load_labelme_polygons(orig_file)
            orig_folder = out_dir / self.saver.sanitize_name(orig_file.stem)
            orig_folder.mkdir(parents=True, exist_ok=True)
            for aug_file in matches:
                file_pairs += 1
                aug_shapes = self.load_labelme_polygons(aug_file)
                m = min(len(orig_shapes), len(aug_shapes))
                pair_folder = orig_folder / self.saver.sanitize_name(aug_file.stem)
                vis_folder = pair_folder / "visualizations"
                vis_folder.mkdir(parents=True, exist_ok=True)
                self.saver.save_text(pair_folder / "shape_summary.txt", self.build_shape_summary_text(orig_file, aug_file, orig_shapes, aug_shapes))
                rows = []
                pair_instances = 0
                pair_cap_sum = 0.0
                pair_cap_bidir_sum = 0.0
                for i in range(m):
                    result, dbg = self.evaluate_shape(i, orig_shapes[i], aug_shapes[i])
                    if result.label_match:
                        pair_instances += 1
                        pair_cap_sum += result.cap_strict
                        pair_cap_bidir_sum += result.cap_bidir
                    vis_path = vis_folder / f"shape_{i:03d}_{self.saver.sanitize_name(result.orig_label)}.png"
                    title = (
                        f"{orig_file.name} | {aug_file.name}\n"
                        f"shape_idx={i} | label={result.orig_label}/{result.aug_label} | recovered={result.recovered_vertices}\n"
                        f"CAP={result.cap_strict:.4f} | CAP_bidir={result.cap_bidir:.4f} | rmse={result.align_rmse:.2f} | thresh={result.distance_threshold:.2f}"
                    )
                    self.visualizer.visualize_one_polygon(
                        dbg["o_pts"], dbg["a_pts"], result.recovered_indices, dbg["aug_to_orig_idx"], dbg["orig_aligned"], vis_path, title
                    )
                    rows.append(
                        [
                            result.shape_idx, result.orig_label, result.aug_label, result.n_points_orig, result.n_points_aug,
                            int(result.label_match), result.recovered_vertices, f"{result.cap_strict:.6f}", f"{result.cap_bidir:.6f}",
                            f"{result.align_rmse:.6f}", f"{result.mean_nn_dist:.6f}" if np.isfinite(result.mean_nn_dist) else "nan",
                            f"{result.distance_threshold:.6f}", " ".join(map(str, result.recovered_indices)), str(vis_path),
                        ]
                    )
                    if result.label_match:
                        plot_records.append(
                            {
                                "cap_strict": result.cap_strict,
                                "cap_bidir": result.cap_bidir,
                                "align_rmse": result.align_rmse,
                                "mean_nn_dist": result.mean_nn_dist,
                                "n_points_orig": result.n_points_orig,
                                "recovered_vertices": result.recovered_vertices,
                                "label": result.orig_label,
                                "visualization_path": str(vis_path),
                                "file_pair": f"{orig_file.stem}->{aug_file.stem}",
                            }
                        )
                self.saver.write_csv(
                    pair_folder / "shape_metrics.csv",
                    ["shape_idx", "orig_label", "rf_label", "n_points_orig", "n_points_rf", "label_match", "recovered_vertices", "cap_strict", "cap_bidir", "align_rmse", "mean_nn_dist", "distance_threshold", "recovered_indices", "visualization_path"],
                    rows,
                )
                if pair_instances == 0:
                    continue
                pair_mean = pair_cap_sum / pair_instances
                pair_bidir = pair_cap_bidir_sum / pair_instances
                total_instances += pair_instances
                total_cap += pair_cap_sum
                total_cap_bidir += pair_cap_bidir_sum
                print(f"{orig_file.name}  <->  {aug_file.name}")
                print(f"instances={pair_instances} | mean_CAP={pair_mean:.4f} | mean_CAP_bidir={pair_bidir:.4f}")
                print("----------------------------------------")
                final_lines.extend([f"{orig_file.name}  <->  {aug_file.name}", f"instances={pair_instances} | mean_CAP={pair_mean:.4f} | mean_CAP_bidir={pair_bidir:.4f}", "----------------------------------------"])
        if total_instances > 0:
            dcap = total_cap / total_instances
            dbi = total_cap_bidir / total_instances
        else:
            dcap = None
            dbi = None
        print("========================================")
        print(f"matched_file_pairs: {file_pairs}")
        print(f"total_instances(N): {total_instances}")
        if dcap is not None:
            print(f"dataset_mean_CAP: {dcap:.6f}")
            print(f"dataset_mean_CAP_bidir: {dbi:.6f}")
        else:
            print("dataset_mean_CAP: n/a (no matched instances)")
        print("========================================")
        final_lines.append("========================================")
        final_lines.append(f"matched_file_pairs: {file_pairs}")
        final_lines.append(f"total_instances(N): {total_instances}")
        if dcap is not None:
            final_lines.append(f"dataset_mean_CAP: {dcap:.6f}")
            final_lines.append(f"dataset_mean_CAP_bidir: {dbi:.6f}")
        else:
            final_lines.append("dataset_mean_CAP: n/a (no matched instances)")
        final_lines.append("========================================")
        self.saver.save_text(out_dir / "final_results.txt", "\n".join(final_lines))
        self.save_paper_plots(plot_records, out_dir, mode_name="roboflow")
        return DatasetEvalResult(file_pairs, total_instances, dcap, dbi)

    def __call__(self, args: argparse.Namespace) -> Optional[DatasetEvalResult]:
        if args.all:
            return self.evaluate_all_pairs(Path(args.orig_dir), Path(args.roboflow_dir), Path(args.out_dir))
        if not args.file:
            raise ValueError("--file is required unless --all is used")
        self.evaluate_single(Path(args.orig_dir), Path(args.roboflow_dir), args.file, args.shape_idx, Path(args.out), args.roboflow_file)
        return None


class YoloProcessor(RoboFlowProcessor):
    """YOLO evaluator using the same geometric matching as Roboflow mode."""

    @staticmethod
    def _canonical_label(label: str) -> str:
        # YOLO export uses bg_area while original annotations use background.
        return "background" if label == "bg_area" else label

    def find_matching_files(self, orig_file: Path, yolo_dir: Path) -> List[Path]:
        base = orig_file.stem
        return sorted([p for p in yolo_dir.glob("*.json") if p.stem.startswith(base + "_")])

    @staticmethod
    def _shape_centroid(shape: Dict) -> np.ndarray:
        pts = np.array(shape["points"], dtype=np.float64)
        return np.mean(pts, axis=0) if len(pts) else np.array([0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _remove_near_duplicates(points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        if len(points) <= 1:
            return points
        out = [points[0]]
        for p in points[1:]:
            if np.linalg.norm(p - out[-1]) > eps:
                out.append(p)
        if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps:
            out.pop()
        return np.array(out, dtype=np.float64)

    @staticmethod
    def _remove_collinear(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        if len(points) < 4:
            return points
        keep = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            p = points[(i - 1) % len(points)]
            c = points[i]
            n = points[(i + 1) % len(points)]
            v1 = c - p
            v2 = n - c
            l1 = float(np.linalg.norm(v1))
            l2 = float(np.linalg.norm(v2))
            if l1 <= eps or l2 <= eps:
                keep[i] = False
                continue
            cross = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
            if cross / (l1 * l2) < 1e-3:
                keep[i] = False
        out = points[keep]
        return out if len(out) >= 3 else points

    def _simplify_augmented_points(self, original_pts: np.ndarray, augmented_pts: np.ndarray) -> np.ndarray:
        pts = self._remove_near_duplicates(augmented_pts)
        pts = self._remove_collinear(pts)
        if len(pts) < 3:
            return augmented_pts
        return pts

    def _match_shape_pairs(self, orig_shapes: List[Dict], aug_shapes: List[Dict]) -> List[Tuple[int, int]]:
        if not orig_shapes or not aug_shapes:
            return []
        orig_c = np.stack([self._shape_centroid(s) for s in orig_shapes], axis=0)
        aug_c = np.stack([self._shape_centroid(s) for s in aug_shapes], axis=0)
        dmat = np.linalg.norm(orig_c[:, None, :] - aug_c[None, :, :], axis=2)
        huge = 1e6
        for i, o in enumerate(orig_shapes):
            ol = self._canonical_label(str(o.get("label", "")))
            for j, a in enumerate(aug_shapes):
                al = self._canonical_label(str(a.get("label", "")))
                if ol != al:
                    dmat[i, j] = huge
        rows, cols = linear_sum_assignment(dmat)
        pairs: List[Tuple[int, int]] = []
        for oi, ai in zip(rows.tolist(), cols.tolist()):
            ol = self._canonical_label(str(orig_shapes[oi].get("label", "")))
            al = self._canonical_label(str(aug_shapes[ai].get("label", "")))
            if ol != al:
                continue
            if dmat[oi, ai] >= huge:
                continue
            pairs.append((oi, ai))
        return pairs

    def evaluate_shape(self, shape_idx: int, orig_shape: Dict, aug_shape: Dict) -> Tuple[ShapeEvalResult, Dict[str, object]]:
        o_pts = np.array(orig_shape["points"], dtype=np.float64)
        a_raw = np.array(aug_shape["points"], dtype=np.float64)
        a_pts = self._simplify_augmented_points(o_pts, a_raw)
        rec_idx, rec_dist, aug_to_orig_idx, orig_aligned, rmse, used = self.recover_indices_aligned_nn(
            o_pts, a_pts, self.distance_threshold
        )
        cap = self.compute_cap_strict(rec_idx, len(o_pts))
        cap_b = self.compute_cap_bidirectional(rec_idx, len(o_pts))
        mean_dist = float(np.mean(rec_dist)) if rec_dist else float("nan")
        result = ShapeEvalResult(
            shape_idx=shape_idx,
            orig_label=str(orig_shape.get("label", "")),
            aug_label=str(aug_shape.get("label", "")),
            n_points_orig=len(o_pts),
            n_points_aug=len(a_pts),
            label_match=self._canonical_label(str(orig_shape.get("label", ""))) == self._canonical_label(str(aug_shape.get("label", ""))),
            recovered_vertices=len(rec_idx),
            cap_strict=cap,
            cap_bidir=cap_b,
            align_rmse=rmse,
            mean_nn_dist=mean_dist,
            distance_threshold=used,
            recovered_indices=rec_idx,
        )
        return result, {"o_pts": o_pts, "a_pts": a_pts, "aug_to_orig_idx": aug_to_orig_idx, "orig_aligned": orig_aligned}

    def build_shape_summary_text(self, orig_file: Path, aug_file: Path, orig_shapes: List[Dict], aug_shapes: List[Dict]) -> str:
        pairs = self._match_shape_pairs(orig_shapes, aug_shapes)
        matched_orig = set(oi for oi, _ in pairs)
        matched_aug = set(ai for _, ai in pairs)
        unmatched_orig: Dict[str, int] = {}
        unmatched_aug: Dict[str, int] = {}
        for i, s in enumerate(orig_shapes):
            if i in matched_orig:
                continue
            lbl = str(s.get("label", ""))
            unmatched_orig[lbl] = unmatched_orig.get(lbl, 0) + 1
        for i, s in enumerate(aug_shapes):
            if i in matched_aug:
                continue
            lbl = str(s.get("label", ""))
            unmatched_aug[lbl] = unmatched_aug.get(lbl, 0) + 1
        lines = [
            "================ SHAPE SUMMARY ================",
            f"Original:  {orig_file.name} | polygon shapes = {len(orig_shapes)}",
            f"YOLO:      {aug_file.name} | polygon shapes = {len(aug_shapes)}",
            f"Matched pairs (label+centroid): {len(pairs)}",
            f"Unmatched original labels: {unmatched_orig if unmatched_orig else '{}'}",
            f"Unmatched YOLO labels: {unmatched_aug if unmatched_aug else '{}'}",
            "------------------------------------------------",
            "pair | orig_idx->yolo_idx | label | n_points_orig | n_points_yolo_raw | n_points_yolo_used",
            "------------------------------------------------",
        ]
        for k, (oi, ai) in enumerate(pairs):
            o_pts = np.array(orig_shapes[oi]["points"], dtype=np.float64)
            a_pts = np.array(aug_shapes[ai]["points"], dtype=np.float64)
            a_used = self._simplify_augmented_points(o_pts, a_pts)
            lines.append(
                f"{k:>4} | {oi:>6}->{ai:<8} | {orig_shapes[oi].get('label','')} | "
                f"{len(orig_shapes[oi]['points']):>12} | {len(aug_shapes[ai]['points']):>17} | {len(a_used):>18}"
            )
        lines.append("================================================")
        return "\n".join(lines)

    def evaluate_single(self, orig_dir: Path, yolo_dir: Path, file_name: str, shape_idx: int, out_path: Path, yolo_file: str = "") -> None:
        orig_file = orig_dir / file_name
        if not orig_file.exists():
            raise FileNotFoundError(f"Original JSON not found: {orig_file}")
        if yolo_file:
            aug_file = yolo_dir / yolo_file
            if not aug_file.exists():
                raise FileNotFoundError(f"Specified YOLO JSON not found: {aug_file}")
        else:
            matches = self.find_matching_files(orig_file, yolo_dir)
            if not matches:
                raise FileNotFoundError(f"No YOLO match found for: {orig_file.name}")
            aug_file = matches[0]
        orig_shapes = self.load_labelme_polygons(orig_file)
        aug_shapes = self.load_labelme_polygons(aug_file)
        pairs = self._match_shape_pairs(orig_shapes, aug_shapes)
        print("\n" + self.build_shape_summary_text(orig_file, aug_file, orig_shapes, aug_shapes))
        print("======== CURRENT JSON EVALUATION ========")
        if shape_idx < 0:
            print("Set --shape-idx to visualize a specific polygon.")
            return
        if shape_idx >= len(pairs):
            raise IndexError(f"shape-idx {shape_idx} out of range (max {len(pairs) - 1})")
        oi, ai = pairs[shape_idx]
        result, dbg = self.evaluate_shape(shape_idx, orig_shapes[oi], aug_shapes[ai])
        title = (
            f"{orig_file.name} | YOLO: {aug_file.name}\n"
            f"pair={shape_idx} (orig={oi}, yolo={ai}) | recovered m={result.recovered_vertices} | CAP={result.cap_strict:.4f}\n"
            f"align_rmse={result.align_rmse:.2f} | mean_nn_dist={result.mean_nn_dist:.2f} | thresh={result.distance_threshold:.2f}"
        )
        self.visualizer.visualize_one_polygon(
            dbg["o_pts"], dbg["a_pts"], result.recovered_indices, dbg["aug_to_orig_idx"], dbg["orig_aligned"], out_path, title
        )
        print(f"Saved: {out_path}")
        print(f"Recovered index sequence: {result.recovered_indices}")
        print(f"CAP (strict): {result.cap_strict:.4f}")
        print(f"CAP (bidirectional): {result.cap_bidir:.4f}")
        print(f"Recovered vertices: {result.recovered_vertices} / {result.n_points_orig}")
        print(f"Alignment RMSE: {result.align_rmse:.4f}")
        print(f"Mean NN distance: {result.mean_nn_dist:.4f}")
        print(f"Distance threshold used: {result.distance_threshold:.4f}")
        print("========================================")

    def evaluate_all_pairs(self, orig_dir: Path, yolo_dir: Path, out_dir: Path) -> DatasetEvalResult:
        orig_files = sorted(orig_dir.glob("*.json"))
        if not orig_files:
            raise FileNotFoundError(f"No JSON files in original dir: {orig_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        total_instances = 0
        total_cap = 0.0
        total_cap_bidir = 0.0
        file_pairs = 0
        plot_records: List[Dict[str, object]] = []
        final_lines: List[str] = [f"Distance threshold: {self.distance_threshold}", "========================================", "======== DATASET CAP EVALUATION ========"]
        print(final_lines[0]); print(final_lines[1]); print(final_lines[2])
        for orig_file in orig_files:
            matches = self.find_matching_files(orig_file, yolo_dir)
            if not matches:
                continue
            orig_shapes = self.load_labelme_polygons(orig_file)
            orig_folder = out_dir / self.saver.sanitize_name(orig_file.stem)
            orig_folder.mkdir(parents=True, exist_ok=True)
            for aug_file in matches:
                file_pairs += 1
                aug_shapes = self.load_labelme_polygons(aug_file)
                pairs = self._match_shape_pairs(orig_shapes, aug_shapes)
                pair_folder = orig_folder / self.saver.sanitize_name(aug_file.stem)
                vis_folder = pair_folder / "visualizations"
                vis_folder.mkdir(parents=True, exist_ok=True)
                self.saver.save_text(pair_folder / "shape_summary.txt", self.build_shape_summary_text(orig_file, aug_file, orig_shapes, aug_shapes))
                rows = []
                pair_instances = 0
                pair_cap_sum = 0.0
                pair_cap_bidir_sum = 0.0
                for pair_idx, (oi, ai) in enumerate(pairs):
                    result, dbg = self.evaluate_shape(pair_idx, orig_shapes[oi], aug_shapes[ai])
                    pair_instances += 1
                    pair_cap_sum += result.cap_strict
                    pair_cap_bidir_sum += result.cap_bidir
                    vis_path = vis_folder / f"pair_{pair_idx:03d}_orig{oi:03d}_yolo{ai:03d}_{self.saver.sanitize_name(result.orig_label)}.png"
                    title = (
                        f"{orig_file.name} | {aug_file.name}\n"
                        f"pair={pair_idx} (orig={oi}, yolo={ai}) | label={result.orig_label}/{result.aug_label} | recovered={result.recovered_vertices}\n"
                        f"CAP={result.cap_strict:.4f} | CAP_bidir={result.cap_bidir:.4f} | rmse={result.align_rmse:.2f} | thresh={result.distance_threshold:.2f}"
                    )
                    self.visualizer.visualize_one_polygon(
                        dbg["o_pts"], dbg["a_pts"], result.recovered_indices, dbg["aug_to_orig_idx"], dbg["orig_aligned"], vis_path, title
                    )
                    rows.append(
                        [
                            pair_idx, oi, ai, result.orig_label, result.aug_label, result.n_points_orig, len(aug_shapes[ai]["points"]), result.n_points_aug,
                            result.recovered_vertices, f"{result.cap_strict:.6f}", f"{result.cap_bidir:.6f}",
                            f"{result.align_rmse:.6f}", f"{result.mean_nn_dist:.6f}" if np.isfinite(result.mean_nn_dist) else "nan",
                            f"{result.distance_threshold:.6f}", " ".join(map(str, result.recovered_indices)), str(vis_path),
                        ]
                    )
                    plot_records.append(
                        {
                            "cap_strict": result.cap_strict,
                            "cap_bidir": result.cap_bidir,
                            "align_rmse": result.align_rmse,
                            "mean_nn_dist": result.mean_nn_dist,
                            "n_points_orig": result.n_points_orig,
                            "recovered_vertices": result.recovered_vertices,
                            "label": result.orig_label,
                            "visualization_path": str(vis_path),
                            "file_pair": f"{orig_file.stem}->{aug_file.stem}",
                        }
                    )
                self.saver.write_csv(
                    pair_folder / "shape_metrics.csv",
                    ["pair_idx", "orig_shape_idx", "yolo_shape_idx", "orig_label", "yolo_label", "n_points_orig", "n_points_yolo_raw", "n_points_yolo_used", "recovered_vertices", "cap_strict", "cap_bidir", "align_rmse", "mean_nn_dist", "distance_threshold", "recovered_indices", "visualization_path"],
                    rows,
                )
                if pair_instances == 0:
                    continue
                pair_mean = pair_cap_sum / pair_instances
                pair_bidir = pair_cap_bidir_sum / pair_instances
                total_instances += pair_instances
                total_cap += pair_cap_sum
                total_cap_bidir += pair_cap_bidir_sum
                print(f"{orig_file.name}  <->  {aug_file.name}")
                print(f"instances={pair_instances} | mean_CAP={pair_mean:.4f} | mean_CAP_bidir={pair_bidir:.4f}")
                print("----------------------------------------")
                final_lines.extend([f"{orig_file.name}  <->  {aug_file.name}", f"instances={pair_instances} | mean_CAP={pair_mean:.4f} | mean_CAP_bidir={pair_bidir:.4f}", "----------------------------------------"])
        if total_instances > 0:
            dcap = total_cap / total_instances
            dbi = total_cap_bidir / total_instances
        else:
            dcap = None
            dbi = None
        print("========================================")
        print(f"matched_file_pairs: {file_pairs}")
        print(f"total_instances(N): {total_instances}")
        if dcap is not None:
            print(f"dataset_mean_CAP: {dcap:.6f}")
            print(f"dataset_mean_CAP_bidir: {dbi:.6f}")
        else:
            print("dataset_mean_CAP: n/a (no matched instances)")
        print("========================================")
        final_lines.append("========================================")
        final_lines.append(f"matched_file_pairs: {file_pairs}")
        final_lines.append(f"total_instances(N): {total_instances}")
        if dcap is not None:
            final_lines.append(f"dataset_mean_CAP: {dcap:.6f}")
            final_lines.append(f"dataset_mean_CAP_bidir: {dbi:.6f}")
        else:
            final_lines.append("dataset_mean_CAP: n/a (no matched instances)")
        final_lines.append("========================================")
        self.saver.save_text(out_dir / "final_results.txt", "\n".join(final_lines))
        self.save_paper_plots(plot_records, out_dir, mode_name="yolo")
        return DatasetEvalResult(file_pairs, total_instances, dcap, dbi)

    def __call__(self, args: argparse.Namespace) -> Optional[DatasetEvalResult]:
        if args.all:
            return self.evaluate_all_pairs(Path(args.orig_dir), Path(args.yolo_dir), Path(args.out_dir))
        if not args.file:
            raise ValueError("--file is required unless --all is used")
        self.evaluate_single(Path(args.orig_dir), Path(args.yolo_dir), args.file, args.shape_idx, Path(args.out), args.yolo_file)
        return None


class SegTOPOAugmentProcessor(BaseCapProcessor):
    """Index-json evaluator for seg-topo-augment outputs."""

    @staticmethod
    def _dedup_preserve_order(seq: List[int]) -> List[int]:
        seen = set()
        out: List[int] = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    @staticmethod
    def _original_points_for_shape(data: Dict[str, object], source_shape_index: int) -> np.ndarray:
        """Load original points (ordered by original index) for one source shape."""
        source_shapes = data.get("source_original_shapes_indexed", [])
        for sh in source_shapes:
            if int(sh.get("source_shape_index", -1)) != source_shape_index:
                continue
            pmap = sh.get("original_points_indexed", {})
            if not pmap:
                break
            ordered = sorted(((int(k), v) for k, v in pmap.items()), key=lambda x: x[0])
            return np.array([[float(v[0]), float(v[1])] for _, v in ordered], dtype=np.float64)
        return np.zeros((0, 2), dtype=np.float64)

    @staticmethod
    def _transformed_points_for_shape(projected_vertices: List[Dict], n_orig: int) -> np.ndarray:
        """Build transformed original-ring points indexed by original vertex id."""
        pts = np.zeros((max(n_orig, 0), 2), dtype=np.float64)
        filled = np.zeros(max(n_orig, 0), dtype=bool)
        for v in projected_vertices:
            oi = int(v.get("original_index", -1))
            tp = v.get("transformed_point", [])
            if oi < 0 or oi >= n_orig or len(tp) != 2:
                continue
            pts[oi] = [float(tp[0]), float(tp[1])]
            filled[oi] = True
        if n_orig <= 0:
            return pts
        # Fallback for any missing transformed points.
        if not np.all(filled):
            for i in range(n_orig):
                if not filled[i]:
                    pts[i] = pts[i - 1] if i > 0 else np.array([0.0, 0.0], dtype=np.float64)
        return pts

    def evaluate_all_indexed(self, indexed_dir: Path, out_dir: Path) -> DatasetEvalResult:
        files = sorted(indexed_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No indexed JSON files found in: {indexed_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        total_instances = 0
        total_cap = 0.0
        total_cap_b = 0.0
        plot_records: List[Dict[str, object]] = []
        final_lines = [
            f"Distance threshold: {self.distance_threshold} (not used in index mode)",
            "========================================",
            "==== SEGTOPO INDEXED CAP EVALUATION ====",
        ]
        print(final_lines[0]); print(final_lines[1]); print(final_lines[2])
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            pshapes = data.get("projected_indexed_shapes", [])
            file_folder = out_dir / self.saver.sanitize_name(fp.stem)
            file_folder.mkdir(parents=True, exist_ok=True)
            vis_folder = file_folder / "visualizations"
            vis_folder.mkdir(parents=True, exist_ok=True)
            rows = []
            f_inst = 0
            f_cap = 0.0
            f_cap_b = 0.0
            for sh in pshapes:
                sid = int(sh.get("source_shape_index", -1))
                label = str(sh.get("label", ""))
                n_orig = int(sh.get("original_vertex_count", 0))
                verts = sh.get("projected_vertices", [])
                transformed_ring = self._transformed_points_for_shape(verts, n_orig)
                comps: Dict[int, List[int]] = {}
                comp_points: Dict[int, List[List[float]]] = {}
                for v in verts:
                    c = int(v.get("projected_component_index", -1))
                    if c < 0:
                        continue
                    oi = int(v.get("original_index", -1))
                    if oi < 0:
                        continue
                    comps.setdefault(c, []).append(oi)
                    pp = v.get("projected_point", [])
                    if len(pp) == 2:
                        comp_points.setdefault(c, []).append([float(pp[0]), float(pp[1])])
                if not comps:
                    rows.append([fp.name, sid, label, -1, n_orig, 0, "0.000000", "0.000000", "", ""])
                    f_inst += 1
                    plot_records.append(
                        {
                            "cap_strict": 0.0,
                            "cap_bidir": 0.0,
                            "align_rmse": float("nan"),
                            "mean_nn_dist": float("nan"),
                            "n_points_orig": n_orig,
                            "recovered_vertices": 0,
                            "label": label,
                            "visualization_path": "",
                            "file_pair": fp.stem,
                        }
                    )
                    continue
                for c, idxs in sorted(comps.items()):
                    indices = self._dedup_preserve_order(idxs)
                    cap = self.compute_cap_strict(indices, n_orig) if n_orig > 0 else 0.0
                    capb = self.compute_cap_bidirectional(indices, n_orig) if n_orig > 0 else 0.0
                    vis_path = ""
                    aug_pts_list = comp_points.get(c, [])
                    if transformed_ring.size > 0 and aug_pts_list and indices:
                        aug_pts = np.array(aug_pts_list, dtype=np.float64)
                        aug_to_orig_idx: List[Optional[int]] = [int(v) for v in idxs[: len(aug_pts)]]
                        vis_out = vis_folder / f"shape_{sid:03d}_{self.saver.sanitize_name(label)}_comp_{c:02d}.png"
                        self.visualizer.visualize_one_polygon(
                            orig_pts=transformed_ring,
                            aug_pts=aug_pts,
                            recovered_idx=indices,
                            aug_to_orig_idx=aug_to_orig_idx,
                            orig_aligned_pts=transformed_ring,
                            out_path=vis_out,
                        )
                        vis_path = str(vis_out)
                    rows.append([fp.name, sid, label, c, n_orig, len(indices), f"{cap:.6f}", f"{capb:.6f}", " ".join(map(str, indices)), vis_path])
                    f_inst += 1
                    f_cap += cap
                    f_cap_b += capb
                    plot_records.append(
                        {
                            "cap_strict": cap,
                            "cap_bidir": capb,
                            "align_rmse": float("nan"),
                            "mean_nn_dist": float("nan"),
                            "n_points_orig": n_orig,
                            "recovered_vertices": len(indices),
                            "label": label,
                            "visualization_path": vis_path,
                            "file_pair": fp.stem,
                        }
                    )
            self.saver.write_csv(
                file_folder / "shape_metrics.csv",
                ["file", "source_shape_index", "label", "component_index", "original_vertex_count", "surviving_vertices", "cap_strict", "cap_bidir", "indices", "visualization_path"],
                rows,
            )
            f_mean = (f_cap / f_inst) if f_inst > 0 else 0.0
            f_mean_b = (f_cap_b / f_inst) if f_inst > 0 else 0.0
            self.saver.save_text(file_folder / "summary.txt", f"file: {fp.name}\ninstances: {f_inst}\nmean_CAP: {f_mean:.6f}\nmean_CAP_bidir: {f_mean_b:.6f}")
            print(f"{fp.name} | instances={f_inst} | mean_CAP={f_mean:.4f} | mean_CAP_bidir={f_mean_b:.4f}")
            print("----------------------------------------")
            final_lines.append(f"{fp.name} | instances={f_inst} | mean_CAP={f_mean:.4f} | mean_CAP_bidir={f_mean_b:.4f}")
            final_lines.append("----------------------------------------")
            total_instances += f_inst
            total_cap += f_cap
            total_cap_b += f_cap_b
        dcap = (total_cap / total_instances) if total_instances > 0 else None
        dbi = (total_cap_b / total_instances) if total_instances > 0 else None
        print("========================================")
        print(f"files: {len(files)}")
        print(f"total_instances(N): {total_instances}")
        if dcap is not None:
            print(f"dataset_mean_CAP: {dcap:.6f}")
            print(f"dataset_mean_CAP_bidir: {dbi:.6f}")
        else:
            print("dataset_mean_CAP: n/a")
        print("========================================")
        final_lines.append("========================================")
        final_lines.append(f"files: {len(files)}")
        final_lines.append(f"total_instances(N): {total_instances}")
        if dcap is not None:
            final_lines.append(f"dataset_mean_CAP: {dcap:.6f}")
            final_lines.append(f"dataset_mean_CAP_bidir: {dbi:.6f}")
        else:
            final_lines.append("dataset_mean_CAP: n/a")
        final_lines.append("========================================")
        self.saver.save_text(out_dir / "final_results.txt", "\n".join(final_lines))
        self.save_paper_plots(plot_records, out_dir, mode_name="segtopo")
        return DatasetEvalResult(len(files), total_instances, dcap, dbi)

    def __call__(self, args: argparse.Namespace) -> Optional[DatasetEvalResult]:
        if not args.all:
            raise ValueError("segtopo mode supports dataset run only; use --all")
        return self.evaluate_all_indexed(Path(args.indexed_dir), Path(args.out_dir))


def build_logger() -> logging.Logger:
    logger = logging.getLogger("cap.main")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["roboflow", "yolo", "segtopo"], default="roboflow", help="Evaluation mode")
    parser.add_argument("--orig-dir", default="seg-topo-augment/json", help="Directory of original LabelMe JSON")
    parser.add_argument("--roboflow-dir", default="roboflow/json", help="Directory of Roboflow LabelMe JSON")
    parser.add_argument("--yolo-dir", default="yolo_train/json", help="Directory of YOLO LabelMe JSON")
    parser.add_argument("--indexed-dir", default="seg-topo-augment/augmented/augmented_index_json", help="Indexed JSON directory for segtopo mode")
    parser.add_argument("--file", help="Original JSON filename (e.g., TE_C_00006.json)")
    parser.add_argument("--roboflow-file", default="", help="Exact Roboflow JSON filename to force current JSON selection")
    parser.add_argument("--yolo-file", default="", help="Exact YOLO JSON filename to force current JSON selection")
    parser.add_argument("--shape-idx", type=int, default=-1, help="Shape index to visualize; -1 prints list only")
    parser.add_argument("--dist-thresh", type=float, default=0.0, help="NN threshold in px; <=0 uses auto scale-aware threshold")
    parser.add_argument("--out", default="cap_debug.png", help="Output image path")
    parser.add_argument("--out-dir", default="", help="Output directory for all-dataset exports")
    parser.add_argument("--all", action="store_true", help="Compute total dataset CAP")
    args = parser.parse_args()

    if not args.out_dir:
        if args.mode == "roboflow":
            args.out_dir = "roboflow/cap_evaluation"
        elif args.mode == "yolo":
            args.out_dir = "yolo_train/cap_evaluation"
        else:
            args.out_dir = "seg-topo-augment/cap_evaluation"

    logger = build_logger()
    visualizer = VisualizationHelper()
    saver = EvaluationDataSaver()

    processor: BaseCapProcessor
    if args.mode == "roboflow":
        processor = RoboFlowProcessor(args.dist_thresh, logger, visualizer, saver)
    elif args.mode == "yolo":
        processor = YoloProcessor(args.dist_thresh, logger, visualizer, saver)
    else:
        processor = SegTOPOAugmentProcessor(args.dist_thresh, logger, visualizer, saver)

    result = processor(args)
    if result is not None:
        logger.info("Completed run: pairs/files=%s, instances=%s", result.matched_file_pairs, result.total_instances)


if __name__ == "__main__":
    main()
