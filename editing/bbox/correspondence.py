import torch
import cv2
import numpy as np

from kornia.feature import LoFTR


class ImageMatcher:

    def __init__(
            self,
            device="cuda"
    ):

        self.device = device

        self.matcher = LoFTR(
            pretrained="outdoor"
        ).to(device)

        self.matcher.eval()


    def preprocess(self, img):

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        tensor = torch.from_numpy(
            gray
        ).float()

        tensor /= 255.0

        tensor = tensor[None,None]

        return tensor.to(self.device)



    @torch.no_grad()
    def match(
            self,
            image0,
            image1
    ):

        """
        image0:
            rendered mesh

        image1:
            Flux image


        Returns:

        pts0:
            coordinates in render image

        pts1:
            coordinates in Flux image

        confidence:
            LoFTR confidence
        """


        img0 = self.preprocess(
            image0
        )

        img1 = self.preprocess(
            image1
        )


        output = self.matcher(
            {
                "image0": img0,
                "image1": img1
            }
        )


        pts0 = (
            output["keypoints0"]
            .cpu()
            .numpy()
        )


        pts1 = (
            output["keypoints1"]
            .cpu()
            .numpy()
        )


        confidence = (
            output["confidence"]
            .cpu()
            .numpy()
        )


        return (
            pts0,
            pts1,
            confidence
        )
    

def visualize_matches(
        img0,
        img1,
        pts0,
        pts1,
        confidence,
        output_path,
        max_matches=100
):

    idx = np.argsort(
        confidence
    )[::-1][:max_matches]


    pts0 = pts0[idx]
    pts1 = pts1[idx]


    h = max(
        img0.shape[0],
        img1.shape[0]
    )

    canvas = np.zeros(
        (
            h,
            img0.shape[1]+img1.shape[1],
            3
        ),
        dtype=np.uint8
    )


    canvas[:img0.shape[0],
           :img0.shape[1]]=img0


    canvas[:img1.shape[0],
           img0.shape[1]:]=img1


    for p0,p1 in zip(pts0,pts1):

        p0=(int(p0[0]),int(p0[1]))

        p1=(
            int(p1[0]+img0.shape[1]),
            int(p1[1])
        )


        cv2.line(
            canvas,
            p0,
            p1,
            (0,255,0),
            1
        )


    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            canvas,
            cv2.COLOR_RGB2BGR
        )
    )

if __name__ == "__main__":


    render_path = "state/render_from_mesh.png"
    flux_path = "state/gen_img_1.png"


    # -------------------------------------------------
    # Load images
    # -------------------------------------------------

    render = cv2.imread(
        render_path
    )

    flux = cv2.imread(
        flux_path
    )


    assert render is not None, \
        f"Cannot load {render_path}"

    assert flux is not None, \
        f"Cannot load {flux_path}"


    # BGR -> RGB

    render = cv2.cvtColor(
        render,
        cv2.COLOR_BGR2RGB
    )


    flux = cv2.cvtColor(
        flux,
        cv2.COLOR_BGR2RGB
    )


    print(
        "Render:",
        render.shape
    )

    print(
        "Flux:",
        flux.shape
    )


    # -------------------------------------------------
    # Matcher
    # -------------------------------------------------

    matcher = ImageMatcher(
        device="cuda"
    )


    # -------------------------------------------------
    # Compute correspondences
    # -------------------------------------------------

    pts0, pts1, conf = matcher.match(
        render,
        flux
    )
    mask = conf > 0.5

    pts0 = pts0[mask]
    pts1 = pts1[mask]
    conf = conf[mask]

    
    print(
        "Matches:",
        len(pts0)
    )

    print("Good matches:", len(pts0))


    print(
        "Mean confidence:",
        conf.mean()
    )


    print(
        "Max confidence:",
        conf.max()
    )


    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    visualize_matches(
        render,
        flux,
        pts0,
        pts1,
        conf,
        "state/matches.png",
        max_matches=200
    )


    print(
        "Saved matches.png"
    )