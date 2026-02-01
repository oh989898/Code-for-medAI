from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .schema import NodeCategory, NodeParam, NodeSchema, PipelineConfig, PipelineEdge


def _node_params_as_dict(params: Dict[str, NodeParam]) -> Dict[str, Any]:
    return {key: value.value for key, value in params.items()}


def _render_data_template(pipeline_name: str, data: Dict[str, Any]) -> str:
    return f'''"""Auto-generated data pipeline for {pipeline_name}."""\n\nfrom pathlib import Path\n\nimport torch\nfrom torch.utils.data import DataLoader\nfrom torchvision import datasets, transforms\n\n\ndef get_dataloaders():\n    data_root = Path(\"{data['data_root']}\")\n    image_size = {data['image_size']}\n    batch_size = {data['batch_size']}\n\n    transform = transforms.Compose(\n        [\n            transforms.Resize((image_size, image_size)),\n            transforms.ToTensor(),\n        ]\n    )\n\n    train_ds = datasets.ImageFolder(data_root / \"train\", transform=transform)\n    val_ds = datasets.ImageFolder(data_root / \"val\", transform=transform)\n\n    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)\n    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)\n\n    return train_loader, val_loader\n'''


def _render_model_template(pipeline_name: str, model: Dict[str, Any]) -> str:
    return f'''"""Auto-generated model definition for {pipeline_name}."""\n\nimport torch\nfrom torch import nn\nfrom torchvision import models\n\n\ndef build_model():\n    model_name = \"{model['architecture']}\"\n    num_classes = {model['num_classes']}\n\n    if not hasattr(models, model_name):\n        raise ValueError(f\"Unknown torchvision model: {{model_name}}\")\n\n    model_fn = getattr(models, model_name)\n    model = model_fn(weights=None)\n\n    if hasattr(model, \"fc\") and isinstance(model.fc, nn.Linear):\n        model.fc = nn.Linear(model.fc.in_features, num_classes)\n    elif hasattr(model, \"classifier\"):\n        if isinstance(model.classifier, nn.Linear):\n            model.classifier = nn.Linear(model.classifier.in_features, num_classes)\n        elif isinstance(model.classifier, nn.Sequential):\n            last_layer = model.classifier[-1]\n            if isinstance(last_layer, nn.Linear):\n                model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)\n    else:\n        raise ValueError(\"Unsupported model head, please customize model.py\")\n\n    return model\n'''


def _render_train_template(pipeline_name: str, train: Dict[str, Any]) -> str:
    return f'''"""Auto-generated training loop for {pipeline_name}."""\n\nfrom pathlib import Path\n\nimport torch\nfrom torch import nn\nfrom torch.optim import Adam\n\nfrom .data import get_dataloaders\nfrom .model import build_model\n\n\ndef train():\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    epochs = {train['epochs']}\n    learning_rate = {train['learning_rate']}\n    output_dir = Path(\"{train['output_dir']}\")\n    output_dir.mkdir(parents=True, exist_ok=True)\n\n    train_loader, val_loader = get_dataloaders()\n    model = build_model().to(device)\n    criterion = nn.CrossEntropyLoss()\n    optimizer = Adam(model.parameters(), lr=learning_rate)\n\n    for epoch in range(1, epochs + 1):\n        model.train()\n        running_loss = 0.0\n        for images, labels in train_loader:\n            images = images.to(device)\n            labels = labels.to(device)\n\n            optimizer.zero_grad()\n            outputs = model(images)\n            loss = criterion(outputs, labels)\n            loss.backward()\n            optimizer.step()\n            running_loss += loss.item()\n\n        avg_loss = running_loss / max(1, len(train_loader))\n        print(f\"Epoch {{epoch}}/{{epochs}} - loss: {{avg_loss:.4f}}\")\n        _validate(model, val_loader, device)\n\n    torch.save(model.state_dict(), output_dir / \"model.pt\")\n\n\ndef _validate(model, loader, device):\n    model.eval()\n    correct = 0\n    total = 0\n    with torch.no_grad():\n        for images, labels in loader:\n            images = images.to(device)\n            labels = labels.to(device)\n            outputs = model(images)\n            _, predicted = torch.max(outputs, 1)\n            total += labels.size(0)\n            correct += (predicted == labels).sum().item()\n\n    accuracy = correct / max(1, total)\n    print(f\"Validation accuracy: {{accuracy:.2%}}\")\n\n\nif __name__ == \"__main__\":\n    train()\n'''


def _render_eval_template(pipeline_name: str, eval_config: Dict[str, Any]) -> str:
    return f'''"""Auto-generated evaluation stub for {pipeline_name}."""\n\nfrom pathlib import Path\n\nimport torch\nfrom torch import nn\n\nfrom .data import get_dataloaders\nfrom .model import build_model\n\n\ndef evaluate():\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    checkpoint_path = Path(\"{eval_config['checkpoint_path']}\")\n\n    _, val_loader = get_dataloaders()\n    model = build_model().to(device)\n    model.load_state_dict(torch.load(checkpoint_path, map_location=device))\n    model.eval()\n\n    correct = 0\n    total = 0\n    with torch.no_grad():\n        for images, labels in val_loader:\n            images = images.to(device)\n            labels = labels.to(device)\n            outputs = model(images)\n            _, predicted = torch.max(outputs, 1)\n            total += labels.size(0)\n            correct += (predicted == labels).sum().item()\n\n    accuracy = correct / max(1, total)\n    print(f\"Evaluation accuracy: {{accuracy:.2%}}\")\n\n\nif __name__ == \"__main__\":\n    evaluate()\n'''


def generate_pipeline(config: PipelineConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data_node = config.get_node(NodeCategory.DATA)
    model_node = config.get_node(NodeCategory.MODEL)
    train_node = config.get_node(NodeCategory.TRAIN)
    eval_node = config.get_node(NodeCategory.EVAL)

    context = {
        "pipeline_name": config.name,
        "data": _node_params_as_dict(data_node.params),
        "model": _node_params_as_dict(model_node.params),
        "train": _node_params_as_dict(train_node.params),
        "eval": _node_params_as_dict(eval_node.params),
    }

    rendered = {
        "data.py": _render_data_template(context["pipeline_name"], context["data"]),
        "model.py": _render_model_template(context["pipeline_name"], context["model"]),
        "train.py": _render_train_template(context["pipeline_name"], context["train"]),
        "eval.py": _render_eval_template(context["pipeline_name"], context["eval"]),
    }

    for filename, content in rendered.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    (output_dir / "__init__.py").write_text("", encoding="utf-8")


def _parse_node(raw_node: Dict[str, Any]) -> NodeSchema:
    params = {
        key: NodeParam(name=value.get("name", key), value=value.get("value"), description=value.get("description"))
        for key, value in raw_node.get("params", {}).items()
    }
    return NodeSchema(
        name=raw_node["name"],
        category=NodeCategory(raw_node["category"]),
        params=params,
        code_template=raw_node.get("code_template"),
    )


def _parse_edges(raw_edges: List[Dict[str, Any]]) -> List[PipelineEdge]:
    return [PipelineEdge(source=edge["source"], target=edge["target"]) for edge in raw_edges]


def load_pipeline_config(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes = [_parse_node(node) for node in raw.get("nodes", [])]
    edges = _parse_edges(raw.get("edges", []))
    return PipelineConfig(
        name=raw.get("name", "medai_pipeline"),
        nodes=nodes,
        edges=edges,
        output_dir=raw.get("output_dir", "generated"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a medical AI pipeline from JSON.")
    parser.add_argument("--config", required=True, help="Path to pipeline JSON file.")
    parser.add_argument("--output", default=None, help="Output directory for generated code.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_pipeline_config(config_path)
    output_dir = Path(args.output) if args.output else Path(config.output_dir)

    generate_pipeline(config, output_dir)
    print(f"Generated pipeline files in: {output_dir}")


if __name__ == "__main__":
    main()
