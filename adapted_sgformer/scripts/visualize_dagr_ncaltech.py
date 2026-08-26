import cv2
import numpy as np
import torch

import yaml
import argparse
from pathlib import Path

from dagr.data.ncaltech101_data import NCaltech101
from dagr.utils.logging import Checkpointer
from dagr.model.networks.dagr import DAGR
from dagr.utils.buffers import format_data
from dagr.model.utils import postprocess_network_output

from torch_geometric.data import Batch

from argparse import Namespace

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def events_to_image(data, scale=2):

    print("pos min:", data.pos.min(dim=0).values)
    print("pos max:", data.pos.max(dim=0).values)
    print("pos shape:", data.pos.shape)
    print(data.pos[:10])

    width = int(data.width)
    height = int(data.height)

    # Fond blanc
    image = np.full(
        (height, width),
        255,
        dtype=np.uint8
    )

    pos = data.pos.detach().cpu().numpy()

    x = pos[:, 0].astype(np.int32)
    y = pos[:, 1].astype(np.int32)

    # Élimine les événements hors image
    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    x = x[valid]
    y = y[valid]

    # Tous les événements en noir
    image[y, x] = 0

    # Passage en BGR pour pouvoir ensuite dessiner
    # les bounding boxes en couleur avec OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Agrandissement uniquement pour l'affichage
    if scale != 1:
        image = cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST
        )

    return image


def draw_ground_truth(image, bbox, scale=2):
    bbox = bbox.detach().cpu().numpy()

    for box in bbox:
        x, y, w, h = box[:4]
        label = int(box[4])

        x1 = int(x * scale)
        y1 = int(y * scale)
        x2 = int((x + w) * scale)
        y2 = int((y + h) * scale)

        # Rectangle GT rouge
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        # Texte GT
        cv2.putText(
            image,
            f"GT {label}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def draw_predictions(
    image,
    prediction,
    width,
    height,
    scale=2,
    conf_threshold=0.28,
):
    boxes = prediction["boxes"].detach().cpu().clone()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()

    # Filtrage confiance
    keep = scores >= conf_threshold

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Les prédictions sont au format [x1, y1, x2, y2]

    # Clip aux limites du capteur
    boxes[:, 0].clamp_(0, width - 1)
    boxes[:, 1].clamp_(0, height - 1)
    boxes[:, 2].clamp_(0, width - 1)
    boxes[:, 3].clamp_(0, height - 1)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()

        if x2 <= x1 or y2 <= y1:
            continue

        pt1 = (
            int(x1 * scale),
            int(y1 * scale),
        )

        pt2 = (
            int(x2 * scale),
            int(y2 * scale),
        )

        # Rectangle prédiction vert
        cv2.rectangle(
            image,
            pt1,
            pt2,
            (0, 255, 0),
            2,
        )

        # Texte prédiction
        text = f"Pred {int(label)} {score:.2f}"

        cv2.putText(
            image,
            text,
            (pt1[0], pt1[1] + 15),  # à l'intérieur de la bbox
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def visualize_prediction(
    model,
    data,
    device="cuda",
    scale=2,
    conf_threshold=0.28,
    show_ground_truth=True,
):
    model.eval()

    # Garde data CPU pour la visualisation
    data_vis = data.clone()

    data_model = Batch.from_data_list([data.clone()])
    
    data_model.to(device)
    data_model = format_data(data_model)

    # Envoie une copie sur GPU pour l'inférence
    # data_model = data.clone().to(device)

    with torch.no_grad():
        out = model(data_model)

    detections = out[0]

    prediction = detections[0]

    image = events_to_image(
        data_vis,
        scale=scale,
    )

    image = draw_ground_truth(
        image,
        data_vis.bbox,
        scale=scale,
    )

    image = draw_predictions(
        image,
        prediction,
        width=int(data_vis.width),
        height=int(data_vis.height),
        scale=scale,
        conf_threshold=conf_threshold,
    )

    return image, prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    checkpoint = torch.load(cfg["model_path"], map_location='cpu', weights_only=False)

    args = checkpoint["args"]

    dataset_path = Path(cfg["data_directory"]) / "ncaltech101"

    dataset = NCaltech101(dataset_path, "test", transform=None, num_events=args["n_nodes"])

    # Exemple :
    #
    # model = ...
    # dataset = ...
    #
    # model.load_state_dict(
    #     torch.load("checkpoint.pt", map_location=device)
    # )
    #
    # model = model.to(device)


    model = DAGR(Namespace(**args), height=dataset.height, width=dataset.width)
    model.load_state_dict(checkpoint['ema'])
    model.eval()
    model.to(device)

    output_dir = Path("visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):

        data = dataset[idx]

        # print(data)

        with torch.no_grad():
            image, prediction = visualize_prediction(
                model=model,
                data=data,
                device=device,
                scale=2,
                conf_threshold=0.28,
                show_ground_truth=True,
            )

        # print(type(prediction))
        # print(prediction)

        # print("Boxes:")
        # print(prediction["boxes"])

        # print("Scores:")
        # print(prediction["scores"])

        # print("Labels:")
        # print(prediction["labels"])

        output_path = output_dir / f"prediction_{idx:05d}.png"

        cv2.imwrite(str(output_path), image)

        print(f"Image enregistrée dans : {output_path}")