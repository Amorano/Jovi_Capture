/**
 * File: node_webcam.js
 * Project: jov_capture
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

const _id = "CAMERA (JOV_CAPTURE)";

app.registerExtension({
	name: 'jov_capture.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            this.addWidget('button', 'SYNC CAMERAS', 'reset', () => {
                console.log("here")
                app.canvas.setDirty(true);
            });
            return me;
        }

        return nodeType;
    }
});
