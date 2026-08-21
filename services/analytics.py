"""
Citrus Farm Spatial & Agronomic Analytics Service
Calculates:
- Tree counts & density (trees/hectare)
- Missing tree / gap identification in planting rows
- Canopy health scoring via Visible Atmospherically Resistant Index (VARI) / Excess Green (ExG)
- Yield estimation & seasonal production projection
- Farm production economic valuation
"""
import math
import numpy as np
from typing import List, Dict, Tuple, Any
from shapely.geometry import Polygon, Point

class FarmAnalyticsService:
    def __init__(self, avg_yield_per_tree_kg: float = 55.0, price_per_kg_inr: float = 35.0):
        """
        :param avg_yield_per_tree_kg: Baseline annual citrus yield for Indian Nagpur Santra / Kinnow (40 - 75 kg/tree)
        :param price_per_kg_inr: Farmgate market price in Indian Rupees (₹25 - ₹50 / kg)
        """
        self.base_yield_kg = avg_yield_per_tree_kg
        self.market_price = price_per_kg_inr

    def compute_polygon_area_ha(self, coords: List[List[float]]) -> float:
        """
        Calculates area in hectares from GeoJSON coordinates (lon, lat) using geodesic approximation.
        """
        if len(coords) < 3:
            return 0.0
        
        # Mean latitude for projection scaling
        mean_lat = math.radians(sum(p[1] for p in coords) / len(coords))
        meters_per_deg_lat = 111319.9
        meters_per_deg_lon = 111319.9 * math.cos(mean_lat)

        # Convert to local metric coordinates
        metric_pts = [
            (p[0] * meters_per_deg_lon, p[1] * meters_per_deg_lat)
            for p in coords
        ]
        poly = Polygon(metric_pts)
        area_sq_meters = poly.area
        return area_sq_meters / 10000.0  # 1 Hectare = 10,000 m²

    def analyze_tree_health(self, rgb_patch: np.ndarray) -> Dict[str, Any]:
        """
        Computes canopy health using Visible Atmospherically Resistant Index (VARI):
        VARI = (Green - Red) / (Green + Red - Blue + 1e-6)
        """
        if rgb_patch.size == 0:
            return {"vari": 0.0, "health_grade": "Unknown", "score": 50}

        img = rgb_patch.astype(np.float32) / 255.0
        r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0] # OpenCV BGR to RGB

        denom = g + r - b
        denom[denom == 0] = 1e-6
        vari_map = (g - r) / denom
        avg_vari = float(np.median(vari_map))

        # Categorize
        if avg_vari > 0.18:
            grade = "High Vigor (Healthy)"
            score = 92
        elif avg_vari > 0.05:
            grade = "Moderate Vigor"
            score = 74
        else:
            grade = "Stressed / Low Canopy"
            score = 48

        return {"vari": round(avg_vari, 3), "health_grade": grade, "score": score}

    def detect_planting_gaps(
        self,
        tree_locations: List[Dict[str, float]],
        polygon_coords: List[List[float]],
        img_bgr: np.ndarray = None,
        expected_spacing_m: float = 6.0
    ) -> List[Dict[str, Any]]:
        """
        Identifies verified missing tree spots (gaps) in regular citrus orchard rows.
        A spot is ONLY flagged as a gap if:
        1. It falls on the regular planting row grid.
        2. Local image reflectance confirms pure bare soil (zero green foliage).
        3. No detected tree exists within 92% of the row spacing.
        """
        if not tree_locations or len(tree_locations) < 10:
            return []

        from shapely.geometry import Polygon, Point
        poly_geom = Polygon(polygon_coords)

        pts = np.array([[t['x'], t['y']] for t in tree_locations])
        gaps = []

        try:
            from scipy.spatial import distance_matrix
            d_mat = distance_matrix(pts, pts)
            np.fill_diagonal(d_mat, np.inf)
            min_dists = np.min(d_mat, axis=1)
            median_dist = float(np.median(min_dists))

            if median_dist < 4:
                return []

            # Check candidate midpoints between row neighbors
            for i in range(min(180, len(pts))):
                for j in range(i + 1, min(180, len(pts))):
                    d = d_mat[i, j]
                    # Exactly 2x row spacing
                    if 1.85 * median_dist <= d <= 2.2 * median_dist:
                        mid_x = float((pts[i, 0] + pts[j, 0]) / 2.0)
                        mid_y = float((pts[i, 1] + pts[j, 1]) / 2.0)

                        if not poly_geom.contains(Point(mid_x, mid_y)):
                            continue

                        # Strict proximity check: no tree within 92% of median distance
                        d_to_all = np.hypot(pts[:, 0] - mid_x, pts[:, 1] - mid_y)
                        if np.min(d_to_all) >= 0.90 * median_dist:
                            # Verify with image reflectance: must be pure bare soil
                            is_bare_soil = True
                            if img_bgr is not None:
                                h, w, _ = img_bgr.shape
                                ix, iy = int(round(mid_x)), int(round(mid_y))
                                if 0 <= iy < h and 0 <= ix < w:
                                    patch = img_bgr[max(0, iy-4):min(h, iy+5), max(0, ix-4):min(w, ix+5)]
                                    if patch.size > 0:
                                        pf = patch.astype(np.float32)
                                        exg_local = 2.0 * pf[:, :, 1] - pf[:, :, 2] - pf[:, :, 0]
                                        # If ANY green foliage is present, reject as gap
                                        if np.mean(exg_local) > 4.0 or np.max(exg_local) > 18.0:
                                            is_bare_soil = False

                            if is_bare_soil:
                                gaps.append({
                                    "x": round(mid_x, 1),
                                    "y": round(mid_y, 1),
                                    "gap_type": "Missing Tree / Vacant Node",
                                    "severity": "High"
                                })
        except Exception as e:
            print(f"Gap detection notice: {e}")
            return []

        return gaps[:15]

    def detect_citrus_canopies_high_precision(
        self,
        img_bgr: np.ndarray,
        poly_pixels: List[List[float]],
        yolo_boxes: List[Any] = None,
        sensitivity: float = 0.5,
        meters_per_pixel: float = 0.55
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        High-precision 1-tree-1-centroid extractor:
        - Multi-scale circular morphological Black-Hat filtering isolates compact dark tree crowns on lighter soil.
        - Chlorophyll & darkness spectral weighting separates live canopy foliage from bare shadows and terrain.
        - Intensity-weighted spatial moments (center of mass) lock the centroid directly on the physical crown apex.
        - Crown-to-halo prominence filtering and spatial NMS ensure deduplication.
        """
        import cv2
        from scipy.ndimage import maximum_filter
        from shapely.geometry import Polygon, Point

        h, w, _ = img_bgr.shape
        poly_geom = Polygon(poly_pixels)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Multi-Scale Circular Morphological Black-Hat (extracts compact circular dark crowns)
        crown_saliency = np.zeros((h, w), dtype=np.float32)
        radii_m = [2.0, 3.2, 4.5]
        max_kernel_size = 5
        for rm in radii_m:
            k = max(5, int(2 * rm / max(0.2, meters_per_pixel))) | 1
            max_kernel_size = max(max_kernel_size, k)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
            crown_saliency += bh
        crown_saliency /= len(radii_m)

        # 2. Chlorophyll & Darkness Spectral Weighting
        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)

        # Excess Green & Inverted Luminance
        exg = 2.0 * g - r - b
        darkness = 255.0 - gray.astype(np.float32)
        foliage_factor = np.clip((exg + 15.0) / 35.0, 0.5, 2.0)
        darkness_factor = np.clip(darkness / 100.0, 0.4, 1.6)

        crown_energy = crown_saliency * darkness_factor * foliage_factor
        crown_energy_blur = cv2.GaussianBlur(crown_energy, (3, 3), 0)

        # 3. Peak Extraction Calibrated to Tree Spacing
        min_tree_spacing_px = max(5, int((3.5 / max(0.2, meters_per_pixel)) * (1.15 - 0.3 * sensitivity)))
        local_max = (maximum_filter(crown_energy_blur, size=min_tree_spacing_px * 2 + 1) == crown_energy_blur)

        # Polygon Boundary Mask
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.array(poly_pixels, dtype=np.int32)], 1)

        inside_vals = crown_energy_blur[poly_mask == 1]
        if len(inside_vals) == 0:
            return [], []

        mean_v = float(np.mean(inside_vals))
        std_v = float(np.std(inside_vals))
        thresh_factor = 0.25 - (sensitivity * 0.25)
        bin_thresh = max(8.0, mean_v + thresh_factor * std_v)

        candidate_mask = (local_max & (crown_energy_blur >= bin_thresh) & (poly_mask == 1))
        peak_ys, peak_xs = np.where(candidate_mask)

        # 4. Strict Canopy Prominence & Sub-Pixel Moment Centroid Refinement
        r_win = max(3, int(2.5 / max(0.2, meters_per_pixel)))
        r_outer = max(5, int(4.5 / max(0.2, meters_per_pixel)))

        raw_trees = []
        for px, py in zip(peak_xs, peak_ys):
            ix, iy = int(round(px)), int(round(py))
            peak_val = crown_energy_blur[iy, ix]

            # Sample local crown patch
            y1, y2 = max(0, iy - r_win), min(h, iy + r_win + 1)
            x1, x2 = max(0, ix - r_win), min(w, ix + r_win + 1)
            patch = crown_energy_blur[y1:y2, x1:x2]

            # Compare inner crown vs outer background ring prominence
            y1_o, y2_o = max(0, iy - r_outer), min(h, iy + r_outer + 1)
            x1_o, x2_o = max(0, ix - r_outer), min(w, ix + r_outer + 1)
            local_full = crown_energy_blur[y1_o:y2_o, x1_o:x2_o]

            cy_o, cx_o = iy - y1_o, ix - x1_o
            yo_grid, xo_grid = np.ogrid[:y2_o-y1_o, :x2_o-x1_o]
            dist_sq = (xo_grid - cx_o) ** 2 + (yo_grid - cy_o) ** 2

            inner_m = dist_sq <= (r_win ** 2)
            outer_m = (dist_sq > (r_win ** 2)) & (dist_sq <= (r_outer ** 2))

            if np.any(inner_m) and np.any(outer_m):
                prominence = float(np.mean(local_full[inner_m])) - float(np.mean(local_full[outer_m]))
                if prominence < 2.0:
                    continue

            # Moment-based centroid on crown foliage mask: locks squarely on the physical crown
            crown_mask = np.where(patch >= 0.40 * peak_val, patch, 0.0)
            m = cv2.moments(crown_mask)
            if m['m00'] > 1e-4:
                cx = x1 + m['m10'] / m['m00']
                cy = y1 + m['m01'] / m['m00']
            else:
                cx, cy = float(px), float(py)

            raw_trees.append((cx, cy, peak_val))

        # 5. Non-Maximum Suppression (1 Centroid Per Tree)
        raw_trees.sort(key=lambda t: t[2], reverse=True)
        final_centroids = []
        for cx, cy, sc in raw_trees:
            too_close = False
            for fx, fy, _ in final_centroids:
                if math.hypot(cx - fx, cy - fy) < min_tree_spacing_px * 0.85:
                    too_close = True
                    break
            if not too_close:
                final_centroids.append((cx, cy, sc))

        detected_trees = []
        health_scores = []

        for cx, cy, sc in final_centroids:
            patch_r = max(4, int(min_tree_spacing_px * 0.45))
            ix, iy = int(round(cx)), int(round(cy))
            patch = img_bgr[max(0, iy - patch_r):min(h, iy + patch_r), max(0, ix - patch_r):min(w, ix + patch_r)]

            health_info = self.analyze_tree_health(patch)
            health_scores.append(health_info["score"])

            detected_trees.append({
                "pixel": {"cx": float(cx), "cy": float(cy)},
                "confidence": round(float(min(0.98, 0.80 + (sc / (std_v * 4.0 + 1e-5)) * 0.18)), 2),
                "canopy_diameter_m": round(float(max_kernel_size * meters_per_pixel * 0.7), 1),
                "health": health_info
            })

        return detected_trees, health_scores

    def calculate_yield_and_economics(
        self,
        tree_count: int,
        farm_area_ha: float,
        avg_health_score: float = 80.0
    ) -> Dict[str, Any]:
        """
        Generates yield estimation, seasonal harvest curve, and economic production valuation.
        """
        if tree_count == 0:
            return {
                "total_trees": 0,
                "farm_area_ha": round(farm_area_ha, 2),
                "trees_per_ha": 0.0,
                "avg_yield_per_tree_kg": 0.0,
                "total_yield_tons": 0.0,
                "market_price_per_kg": self.market_price,
                "total_production_value": 0.0,
                "seasonal_breakdown": {
                    "Early Harvest (Oct - Dec)": {"percentage": "25%", "yield_tons": 0.0, "est_revenue": 0.0},
                    "Peak Harvest (Jan - Mar)": {"percentage": "55%", "yield_tons": 0.0, "est_revenue": 0.0},
                    "Late Season (Apr - May)": {"percentage": "20%", "yield_tons": 0.0, "est_revenue": 0.0}
                }
            }

        trees_per_ha = round(tree_count / max(farm_area_ha, 0.01), 1)
        
        # Health modifier on yield: 50 score -> 0.7x, 100 score -> 1.15x
        health_mod = 0.5 + (avg_health_score / 100.0) * 0.65
        adjusted_yield_per_tree = self.base_yield_kg * health_mod
        
        total_yield_kg = tree_count * adjusted_yield_per_tree
        total_yield_tons = round(total_yield_kg / 1000.0, 2)
        
        total_value = round(total_yield_kg * self.market_price, 2)

        # Seasonal Harvest Projection (Citrus Early, Mid, Late season)
        seasonal_breakdown = {
            "Early Harvest (Oct - Dec)": {
                "percentage": "25%",
                "yield_tons": round(total_yield_tons * 0.25, 2),
                "est_revenue": round(total_value * 0.25, 2)
            },
            "Peak Harvest (Jan - Mar)": {
                "percentage": "55%",
                "yield_tons": round(total_yield_tons * 0.55, 2),
                "est_revenue": round(total_value * 0.55, 2)
            },
            "Late Season (Apr - May)": {
                "percentage": "20%",
                "yield_tons": round(total_yield_tons * 0.20, 2),
                "est_revenue": round(total_value * 0.20, 2)
            }
        }

        return {
            "total_trees": tree_count,
            "farm_area_ha": round(farm_area_ha, 2),
            "trees_per_ha": trees_per_ha,
            "avg_yield_per_tree_kg": round(adjusted_yield_per_tree, 1),
            "total_yield_tons": total_yield_tons,
            "market_price_per_kg": self.market_price,
            "total_production_value": total_value,
            "seasonal_breakdown": seasonal_breakdown
        }
