const paletteData = [
  {
    id: "data",
    name: "数据节点",
    category: "data",
    params: {
      data_root: "./data",
      image_size: 224,
      batch_size: 16,
    },
  },
  {
    id: "model",
    name: "模型节点",
    category: "model",
    params: {
      architecture: "resnet18",
      num_classes: 2,
    },
  },
  {
    id: "train",
    name: "训练节点",
    category: "train",
    params: {
      epochs: 5,
      learning_rate: 0.001,
      output_dir: "./outputs",
    },
  },
  {
    id: "eval",
    name: "评估节点",
    category: "eval",
    params: {
      checkpoint_path: "./outputs/model.pt",
    },
  },
];

const state = {
  nodes: [],
  selectedId: null,
};

const palette = document.getElementById("palette");
const canvas = document.getElementById("canvas");
const inspector = document.getElementById("inspector");
const output = document.getElementById("output");

function createPalette() {
  paletteData.forEach((node) => {
    const item = document.createElement("div");
    item.className = "palette-item";
    item.draggable = true;
    item.dataset.nodeId = node.id;
    item.innerHTML = `<strong>${node.name}</strong><span>${node.category}</span>`;
    item.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("nodeId", node.id);
    });
    palette.appendChild(item);
  });
}

function addNode(nodeId, x, y) {
  const base = paletteData.find((node) => node.id === nodeId);
  if (!base) return;

  const instanceId = `${nodeId}-${Date.now()}`;
  const newNode = {
    instanceId,
    name: base.name,
    category: base.category,
    params: { ...base.params },
    position: { x, y },
  };

  state.nodes.push(newNode);
  renderNodes();
  updateOutput();
  selectNode(instanceId);
}

function renderNodes() {
  canvas.innerHTML = "";
  state.nodes.forEach((node) => {
    const card = document.createElement("div");
    card.className = "node";
    if (state.selectedId === node.instanceId) {
      card.classList.add("selected");
    }
    card.style.left = `${node.position.x}px`;
    card.style.top = `${node.position.y}px`;
    card.dataset.nodeInstanceId = node.instanceId;
    card.innerHTML = `<h3>${node.name}</h3><p>${node.category}</p>`;

    card.addEventListener("mousedown", (event) => startDrag(event, node.instanceId));
    card.addEventListener("click", () => selectNode(node.instanceId));

    canvas.appendChild(card);
  });
}

function selectNode(instanceId) {
  state.selectedId = instanceId;
  renderNodes();
  renderInspector();
}

function renderInspector() {
  inspector.innerHTML = "";
  const node = state.nodes.find((item) => item.instanceId === state.selectedId);
  if (!node) {
    inspector.innerHTML = '<p class="muted">选择一个节点以编辑参数。</p>';
    return;
  }

  const title = document.createElement("h3");
  title.textContent = node.name;
  inspector.appendChild(title);

  Object.entries(node.params).forEach(([key, value]) => {
    const label = document.createElement("label");
    label.textContent = key;
    const input = document.createElement("input");
    input.value = value;
    input.addEventListener("input", (event) => {
      node.params[key] = parseValue(event.target.value);
      updateOutput();
    });
    label.appendChild(input);
    inspector.appendChild(label);
  });
}

function parseValue(raw) {
  if (raw === "true") return true;
  if (raw === "false") return false;
  const number = Number(raw);
  if (!Number.isNaN(number) && raw.trim() !== "") return number;
  return raw;
}

function updateOutput() {
  const nodes = state.nodes.map((node) => ({
    name: node.name,
    category: node.category,
    params: Object.entries(node.params).reduce((acc, [key, value]) => {
      acc[key] = { name: key, value };
      return acc;
    }, {}),
  }));

  const edges = buildEdges(nodes);

  const payload = {
    name: "ui_pipeline",
    output_dir: "generated",
    nodes,
    edges,
  };

  output.value = JSON.stringify(payload, null, 2);
}

function buildEdges(nodes) {
  const edges = [];
  const categoryOrder = ["data", "model", "train", "eval"];
  const latestByCategory = {};

  categoryOrder.forEach((category) => {
    const node = nodes.find((item) => item.category === category);
    if (node) {
      latestByCategory[category] = node.name;
    }
  });

  if (latestByCategory.data && latestByCategory.train) {
    edges.push({ source: latestByCategory.data, target: latestByCategory.train });
  }
  if (latestByCategory.model && latestByCategory.train) {
    edges.push({ source: latestByCategory.model, target: latestByCategory.train });
  }
  if (latestByCategory.train && latestByCategory.eval) {
    edges.push({ source: latestByCategory.train, target: latestByCategory.eval });
  }

  return edges;
}

function startDrag(event, instanceId) {
  event.preventDefault();
  const node = state.nodes.find((item) => item.instanceId === instanceId);
  if (!node) return;

  const startX = event.clientX;
  const startY = event.clientY;
  const originX = node.position.x;
  const originY = node.position.y;

  function onMove(moveEvent) {
    const deltaX = moveEvent.clientX - startX;
    const deltaY = moveEvent.clientY - startY;
    node.position.x = Math.max(0, originX + deltaX);
    node.position.y = Math.max(0, originY + deltaY);
    renderNodes();
  }

  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    updateOutput();
  }

  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

canvas.addEventListener("dragover", (event) => {
  event.preventDefault();
});

canvas.addEventListener("drop", (event) => {
  event.preventDefault();
  const nodeId = event.dataTransfer.getData("nodeId");
  const rect = canvas.getBoundingClientRect();
  addNode(nodeId, event.clientX - rect.left, event.clientY - rect.top);
});

const clearBtn = document.getElementById("clear-btn");
clearBtn.addEventListener("click", () => {
  state.nodes = [];
  state.selectedId = null;
  renderNodes();
  renderInspector();
  updateOutput();
});

const exportBtn = document.getElementById("export-btn");
exportBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(output.value);
  exportBtn.textContent = "已复制";
  setTimeout(() => {
    exportBtn.textContent = "导出 JSON";
  }, 1500);
});

createPalette();
updateOutput();
renderInspector();
