# view_search.py

import os
import cv2
import numpy as np
import torch

from editing.bbox.render import render_single_view_for_bbox
from editing.bbox.correspondence import ImageMatcher, visualize_matches


# ============================================================
# Configuration
# ============================================================

WIDTH = 518
HEIGHT = 518

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "state"

# LoFTR filtering
CONFIDENCE_THRESHOLD = 0.5

# Homography RANSAC threshold in pixels
RANSAC_THRESHOLD = 5.0

# ------------------------------------------------------------
# COARSE SEARCH
# ------------------------------------------------------------

COARSE_AZIMUTH_STEP = 30
COARSE_ELEVATIONS = [-30, -10, 10, 30]

# ------------------------------------------------------------
# FINE SEARCH
# ------------------------------------------------------------

FINE_AZIMUTH_RADIUS = 30
FINE_ELEVATION_RADIUS = 20

FINE_AZIMUTH_STEP = 5
FINE_ELEVATION_STEP = 5


# ============================================================
# Utility
# ============================================================

def ensure_output_dir():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# Image loading
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    return image


# ============================================================
# View scoring
# ============================================================

def score_view(
        render,
        flux,
        matcher,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        ransac_threshold=RANSAC_THRESHOLD
):
    """
    Compute the correspondence quality between:

        render = candidate mesh rendering
        flux   = Flux image

    Returns a dictionary containing:

        score
        number of matches
        number of RANSAC inliers
        mean LoFTR confidence
        median reprojection error
        homography
        correspondence points
    """

    pts_render, pts_flux, confidence = matcher.match(
        render,
        flux
    )

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    confidence_mask = (
        confidence >= confidence_threshold
    )

    pts_render = pts_render[confidence_mask]
    pts_flux = pts_flux[confidence_mask]
    confidence = confidence[confidence_mask]

    num_matches = len(pts_render)

    if num_matches < 4:

        return {
            "score": 0.0,
            "matches": num_matches,
            "inliers": 0,
            "mean_confidence": 0.0,
            "median_error": np.inf,
            "H": None,
            "pts_render": pts_render,
            "pts_flux": pts_flux,
            "confidence": confidence
        }

    # --------------------------------------------------------
    # Estimate homography using RANSAC
    # --------------------------------------------------------

    H, inlier_mask = cv2.findHomography(
        pts_flux,
        pts_render,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )

    if H is None:

        return {
            "score": 0.0,
            "matches": num_matches,
            "inliers": 0,
            "mean_confidence": 0.0,
            "median_error": np.inf,
            "H": None,
            "pts_render": pts_render,
            "pts_flux": pts_flux,
            "confidence": confidence
        }

    if inlier_mask is None:

        return {
            "score": 0.0,
            "matches": num_matches,
            "inliers": 0,
            "mean_confidence": 0.0,
            "median_error": np.inf,
            "H": H,
            "pts_render": pts_render,
            "pts_flux": pts_flux,
            "confidence": confidence
        }

    inlier_mask = (
        inlier_mask.ravel().astype(bool)
    )

    num_inliers = int(
        np.sum(inlier_mask)
    )

    if num_inliers < 4:

        return {
            "score": 0.0,
            "matches": num_matches,
            "inliers": num_inliers,
            "mean_confidence": 0.0,
            "median_error": np.inf,
            "H": H,
            "pts_render": pts_render,
            "pts_flux": pts_flux,
            "confidence": confidence
        }

    # --------------------------------------------------------
    # Reprojection error
    # --------------------------------------------------------

    flux_inliers = (
        pts_flux[inlier_mask]
    )

    render_inliers = (
        pts_render[inlier_mask]
    )

    projected = cv2.perspectiveTransform(
        flux_inliers
        .reshape(-1, 1, 2)
        .astype(np.float32),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - render_inliers,
        axis=1
    )

    median_error = float(
        np.median(errors)
    )

    mean_confidence = float(
        np.mean(
            confidence[inlier_mask]
        )
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------
    #
    # High number of inliers is good.
    # High confidence is good.
    # Low reprojection error is good.
    #
    # --------------------------------------------------------

    score = (
        num_inliers
        * mean_confidence
        / (1.0 + median_error)
    )

    return {
        "score": float(score),
        "matches": num_matches,
        "inliers": num_inliers,
        "mean_confidence": mean_confidence,
        "median_error": median_error,
        "H": H,
        "pts_render": pts_render,
        "pts_flux": pts_flux,
        "confidence": confidence
    }


# ============================================================
# Save best-view correspondence visualization
# ============================================================

def save_best_matches(
        render,
        flux,
        result,
        output_path
):
    """
    Save LoFTR correspondences for the selected view.
    """

    pts_render = result["pts_render"]
    pts_flux = result["pts_flux"]
    confidence = result["confidence"]

    if len(pts_render) == 0:
        return

    visualize_matches(
        render,
        flux,
        pts_render,
        pts_flux,
        confidence,
        output_path,
        max_matches=200
    )


# ============================================================
# Save render
# ============================================================

def save_rgb_image(
        image,
        path
):

    cv2.imwrite(
        path,
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    )


# ============================================================
# Search
# ============================================================

def search_best_view(
        mesh_path,
        flux_image,
        fx, fy, cx, cy,
        matcher,
        azimuths,
        elevations,
        stage_name="search"
):
    """
    Search candidate camera poses and return the best one.
    """

    best_result = None
    all_results = []

    total = (
        len(azimuths)
        * len(elevations)
    )

    counter = 0

    print()
    print("=" * 70)
    print(f"{stage_name.upper()} VIEW SEARCH")
    print(f"Testing {total} candidate views")
    print("=" * 70)

    for elevation in elevations:

        for azimuth in azimuths:

            counter += 1

            # print()
            # print(
            #     f"[{counter}/{total}] "
            #     f"azimuth={azimuth:.1f}° "
            #     f"elevation={elevation:.1f}°"
            # )

            # ------------------------------------------------
            # Convert degrees -> radians
            # ------------------------------------------------

            azimuth_rad = np.radians(
                azimuth
            )

            elevation_rad = np.radians(
                elevation
            )

            # ------------------------------------------------
            # Render candidate view
            # ------------------------------------------------

            (
                render,
                depth,
                camera_pose,
                scale,
                center
            ) = render_single_view_for_bbox(
                mesh_path,
                fx, fy, cx, cy,
                azimuth=azimuth_rad,
                elevation=elevation_rad
            )

            # ------------------------------------------------
            # Score correspondence
            # ------------------------------------------------

            result = score_view(
                render,
                flux_image,
                matcher
            )

            # ------------------------------------------------
            # Store camera information
            # ------------------------------------------------

            result["azimuth"] = azimuth
            result["elevation"] = elevation

            result["azimuth_rad"] = azimuth_rad
            result["elevation_rad"] = elevation_rad

            result["render"] = render
            result["depth"] = depth

            result["camera_pose"] = camera_pose
            result["scale"] = scale
            result["center"] = center

            all_results.append(
                result
            )

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            # print(
            #     f"    matches       : "
            #     f"{result['matches']}"
            # )

            # print(
            #     f"    RANSAC inliers: "
            #     f"{result['inliers']}"
            # )

            # print(
            #     f"    confidence    : "
            #     f"{result['mean_confidence']:.4f}"
            # )

            # if np.isfinite(
            #     result["median_error"]
            # ):

            #     print(
            #         f"    median error  : "
            #         f"{result['median_error']:.3f}px"
            #     )

            # else:

            #     print(
            #         "    median error  : inf"
            #     )

            # print(
            #     f"    SCORE         : "
            #     f"{result['score']:.4f}"
            # )

            # ------------------------------------------------
            # Keep best
            # ------------------------------------------------

            if (
                best_result is None
                or
                result["score"]
                >
                best_result["score"]
            ):

                best_result = result

                # print()
                # print(
                #     "    *** NEW BEST VIEW ***"
                # )

    return (
        best_result,
        all_results
    )


# ============================================================
# Fine search range
# ============================================================

def create_fine_search_ranges(
        coarse_best
):
    """
    Create a fine search around the coarse winner.
    """

    best_az = (
        coarse_best["azimuth"]
    )

    best_el = (
        coarse_best["elevation"]
    )

    az_min = (
        best_az
        - FINE_AZIMUTH_RADIUS
    )

    az_max = (
        best_az
        + FINE_AZIMUTH_RADIUS
    )

    el_min = (
        best_el
        - FINE_ELEVATION_RADIUS
    )

    el_max = (
        best_el
        + FINE_ELEVATION_RADIUS
    )

    azimuths = np.arange(
        az_min,
        az_max + FINE_AZIMUTH_STEP,
        FINE_AZIMUTH_STEP
    )

    elevations = np.arange(
        el_min,
        el_max + FINE_ELEVATION_STEP,
        FINE_ELEVATION_STEP
    )

    return (
        azimuths,
        elevations
    )


# ============================================================
# Save search results
# ============================================================

def save_results_table(
        results,
        path
):

    results_sorted = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "rank,azimuth,elevation,"
            "score,matches,inliers,"
            "confidence,median_error\n"
        )

        for rank, result in enumerate(
            results_sorted,
            start=1
        ):

            error = result[
                "median_error"
            ]

            if not np.isfinite(error):
                error = -1

            f.write(
                f"{rank},"
                f"{result['azimuth']:.2f},"
                f"{result['elevation']:.2f},"
                f"{result['score']:.6f},"
                f"{result['matches']},"
                f"{result['inliers']},"
                f"{result['mean_confidence']:.6f},"
                f"{error:.6f}\n"
            )


# ============================================================
# The end2end method
# ============================================================

def find_best_angles(mesh_path, flux_path, fx, fy, cx, cy):

    ensure_output_dir()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not os.path.exists(
        mesh_path
    ):

        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}"
        )

    if not os.path.exists(
        flux_path
    ):

        raise FileNotFoundError(
            f"Flux image not found: {flux_path}"
        )

    # --------------------------------------------------------
    # Load Flux image
    # --------------------------------------------------------

    print(
        "Loading Flux image..."
    )

    flux = load_image(
        flux_path
    )

    print(
        "Flux shape:",
        flux.shape
    )

    # --------------------------------------------------------
    # Load LoFTR
    # --------------------------------------------------------

    print()
    print(
        f"Loading LoFTR on {DEVICE}..."
    )

    matcher = ImageMatcher(
        device=DEVICE
    )

    print(
        "LoFTR loaded."
    )

    # ========================================================
    # COARSE SEARCH
    # ========================================================
    print()
    print("Coarse research starting...")

    coarse_azimuths = np.arange(
        0,
        360,
        COARSE_AZIMUTH_STEP
    )

    coarse_elevations = (
        COARSE_ELEVATIONS
    )

    coarse_best, coarse_results = (
        search_best_view(
            mesh_path=mesh_path,
            flux_image=flux,
            fx=fx, fy=fy, cx=cx, cy=cy,
            matcher=matcher,
            azimuths=coarse_azimuths,
            elevations=coarse_elevations,
            stage_name="coarse"
        )
    )

    if coarse_best is None:
        raise RuntimeError(
            "Coarse search failed."
        )

    # --------------------------------------------------------
    # Print coarse winner
    # --------------------------------------------------------

    print()
    print("Coarse research finished!")

    # --------------------------------------------------------
    # Save coarse winner
    # --------------------------------------------------------

    # save_rgb_image(
    #     coarse_best["render"],
    #     os.path.join(
    #         OUTPUT_DIR,
    #         "coarse_best_render.png"
    #     )
    # )

    # save_best_matches(
    #     coarse_best["render"],
    #     flux,
    #     coarse_best,
    #     os.path.join(
    #         OUTPUT_DIR,
    #         "coarse_best_matches.png"
    #     )
    # )

    # save_results_table(
    #     coarse_results,
    #     os.path.join(
    #         OUTPUT_DIR,
    #         "coarse_results.csv"
    #     )
    # )

    # ========================================================
    # FINE SEARCH
    # ========================================================

    print()
    print("Fine research starting...")

    (
        fine_azimuths,
        fine_elevations
    ) = create_fine_search_ranges(
        coarse_best
    )

    fine_best, fine_results = (
        search_best_view(
            mesh_path=mesh_path,
            flux_image=flux,
            fx=fx, fy=fy, cx=cx, cy=cy,
            matcher=matcher,
            azimuths=fine_azimuths,
            elevations=fine_elevations,
            stage_name="fine"
        )
    )

    if fine_best is None:
        raise RuntimeError(
            "Fine search failed."
        )

    print()
    print("Fine research finished!")

    # ========================================================
    # FINAL RESULT
    # ========================================================

    # print()
    # print("=" * 70)
    # print("FINAL BEST VIEW")
    # print("=" * 70)

    # print(
    #     f"Azimuth   : "
    #     f"{fine_best['azimuth']:.2f}°"
    # )

    # print(
    #     f"Elevation : "
    #     f"{fine_best['elevation']:.2f}°"
    # )

    # print(
    #     f"Score     : "
    #     f"{fine_best['score']:.4f}"
    # )

    # print(
    #     f"Matches   : "
    #     f"{fine_best['matches']}"
    # )

    # print(
    #     f"Inliers   : "
    #     f"{fine_best['inliers']}"
    # )

    # print(
    #     f"Confidence: "
    #     f"{fine_best['mean_confidence']:.4f}"
    # )

    # print(
    #     f"Error     : "
    #     f"{fine_best['median_error']:.3f}px"
    # )

    # # --------------------------------------------------------
    # # Save final render
    # # --------------------------------------------------------

    final_render_path = os.path.join(
        OUTPUT_DIR,
        "render_from_mesh.png"
    )

    save_rgb_image(
        fine_best["render"],
        final_render_path
    )

    # # --------------------------------------------------------
    # # Save final correspondence
    # # --------------------------------------------------------

    final_matches_path = os.path.join(
        OUTPUT_DIR,
        "best_view_matches.png"
    )

    save_best_matches(
        fine_best["render"],
        flux,
        fine_best,
        final_matches_path
    )

    # # --------------------------------------------------------
    # # Save final results table
    # # --------------------------------------------------------

    # save_results_table(
    #     fine_results,
    #     os.path.join(
    #         OUTPUT_DIR,
    #         "fine_results.csv"
    #     )
    # )

    # --------------------------------------------------------
    # Save final homography
    # --------------------------------------------------------

    if fine_best["H"] is not None:

        np.save(
            os.path.join(
                OUTPUT_DIR,
                "best_homography.npy"
            ),
            fine_best["H"]
        )

    # --------------------------------------------------------
    # Save camera parameters
    # --------------------------------------------------------

    # camera_info_path = os.path.join(
    #     OUTPUT_DIR,
    #     "best_camera.txt"
    # )

    # with open(
    #     camera_info_path,
    #     "w"
    # ) as f:

    #     f.write(
    #         f"azimuth_degrees="
    #         f"{fine_best['azimuth']}\n"
    #     )

    #     f.write(
    #         f"elevation_degrees="
    #         f"{fine_best['elevation']}\n"
    #     )

    #     f.write(
    #         f"azimuth_radians="
    #         f"{fine_best['azimuth_rad']}\n"
    #     )

    #     f.write(
    #         f"elevation_radians="
    #         f"{fine_best['elevation_rad']}\n"
    #     )

    #     f.write(
    #         f"score="
    #         f"{fine_best['score']}\n"
    #     )

    #     f.write(
    #         f"matches="
    #         f"{fine_best['matches']}\n"
    #     )

    #     f.write(
    #         f"inliers="
    #         f"{fine_best['inliers']}\n"
    #     )

    #     f.write(
    #         f"mean_confidence="
    #         f"{fine_best['mean_confidence']}\n"
    #     )

    #     f.write(
    #         f"median_error="
    #         f"{fine_best['median_error']}\n"
    #     )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 70)
    print("SEARCH FINISHED")
    print("=" * 70)

    print(
        "Best render:"
    )

    print(
        final_render_path
    )

    print()
    print(
        "Best correspondences:"
    )

    print(
        final_matches_path
    )

    # print()
    # print(
    #     "Best camera:"
    # )

    # print(
    #     camera_info_path
    # )

    print()
    print(
        "The selected viewpoint is:"
    )

    print(
        f"  azimuth   = "
        f"{fine_best['azimuth']:.2f}°"
    )

    print(
        f"  elevation = "
        f"{fine_best['elevation']:.2f}°"
    )

    return (
        fine_best['H'],
        fine_best['camera_pose'],
        fine_best['depth'],
        fine_best['center'],
        fine_best['scale']
    )


# if __name__ == "__main__":
#     H, camera_pose, depth_map, center,scale = find_best_angles("mesh_umbr.glb", "gen_img_1.png",1209.865588803089, 1195.9805743243244, 256.0, 256.0)
#     print(H)
#     print(camera_pose)
#     print(depth_map)
#     print(center)
#     print(scale)