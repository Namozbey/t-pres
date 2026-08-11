import cv2
from editing.bbox.correspondence import ImageMatcher


def estimate_homography(
        pts_render,
        pts_flux,
        confidence=None,
        confidence_threshold=0.4,
        ransac_threshold=5.0
):
    """
    Estimate mapping:
        Flux image
              |
              v
        Render image
    pts_render:
        coordinates in rendered mesh image
    pts_flux:
        coordinates in Flux image
    """
    # ---------------------------------------------
    # confidence filtering
    # ---------------------------------------------
    if confidence is not None:
        mask = confidence > confidence_threshold
        pts_render = pts_render[mask]
        pts_flux = pts_flux[mask]

    print(
        "Using matches:",
        len(pts_render)
    )

    if len(pts_render) < 4:
        raise RuntimeError(
            "Not enough matches for homography"
        )
    # ---------------------------------------------
    # Homography
    # ---------------------------------------------
    H, inliers = cv2.findHomography(
        pts_flux,
        pts_render,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )

    if H is None:
        raise RuntimeError(
            "Homography estimation failed"
        )

    print(
        "Homography inliers:",
        inliers.sum(),
        "/",
        len(inliers)
    )

    return H, inliers

def warp_difference(
        difference,
        H,
        output_size
):
    """
    Warp difference image from Flux space
    into render space.
    output_size:
        (width,height)
    """
    aligned = cv2.warpPerspective(
        difference,
        H,
        output_size,
        flags=cv2.INTER_LINEAR
    )

    return aligned

def save_debug(
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

if __name__ == "__main__":
    # ---------------------------------------------
    # Example standalone test
    # ---------------------------------------------
    render = cv2.imread(
        "state/render_from_mesh.png"
    )

    flux = cv2.imread(
        "state/gen_img_1.png"
    )

    diff = cv2.imread(
        "state/changed_part.png"
    )

    render = cv2.cvtColor(
        render,
        cv2.COLOR_BGR2RGB
    )

    flux = cv2.cvtColor(
        flux,
        cv2.COLOR_BGR2RGB
    )

    diff = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2RGB
    )

    matcher = ImageMatcher()

    pts_render, pts_flux, conf = matcher.match(
        render,
        flux
    )

    H, inliers = estimate_homography(
        pts_render,
        pts_flux,
        conf
    )

    aligned = warp_difference(
        diff,
        H,
        (
            render.shape[1],
            render.shape[0]
        )
    )

    save_debug(
        aligned,
        "state/aligned_difference.png"
    )

    print(
        "Saved aligned_difference.png"
    )