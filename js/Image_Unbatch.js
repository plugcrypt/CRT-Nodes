import { app } from "/scripts/app.js";

app.registerExtension({
  name: "CRT.ImageUnbatch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CRTImageUnbatch") {
      return;
    }

    const MAX_IMAGES = 64;

    const syncSize = (node) => {
      node.size = node.computeSize();
      node.setDirtyCanvas(true, true);
    };

    const syncOutputs = (node) => {
      if (!node.outputs || node.outputs.length === 0) {
        node.addOutput("image_1", "IMAGE");
      }

      while (node.outputs.length > 1) {
        const lastIdx = node.outputs.length - 1;
        const prevIdx = node.outputs.length - 2;
        const lastEmpty = (node.outputs[lastIdx].links || []).length === 0;
        const prevEmpty = (node.outputs[prevIdx].links || []).length === 0;
        if (lastEmpty && prevEmpty) {
          node.removeOutput(lastIdx);
        } else {
          break;
        }
      }

      const lastIdx = node.outputs.length - 1;
      const lastConnected = (node.outputs[lastIdx].links || []).length > 0;
      if (node.outputs.length < MAX_IMAGES && lastConnected) {
        node.addOutput(`image_${node.outputs.length + 1}`, "IMAGE");
      }

      syncSize(node);
    };

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = originalOnNodeCreated?.apply(this, arguments);
      syncOutputs(this);
      return r;
    };

    const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (slotType) {
      const r = originalOnConnectionsChange?.apply(this, arguments);
      if (slotType === LiteGraph.OUTPUT) {
        syncOutputs(this);
      }
      return r;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = originalOnConfigure?.apply(this, arguments);
      syncOutputs(this);
      return r;
    };
  },
});