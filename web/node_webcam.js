/**
 * File: node_webcam.js
 * Project: jov_capture
 */

import { app } from "../../../scripts/app.js";
import { api_get } from './util_jov.js'

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

            const widget_camera = this.widgets.find(w => w.name == 'CAMERA');

            this.addWidget('button', 'REFRESH CAMERA LIST', 'refresh', async () => {
                var data = await api_get("/jov_capture/camera");
                widget_camera.options.values = data;
                widget_camera.value = data[0];
                console.info(widget_camera)
                app.canvas.setDirty(true);
            });
            return me;
        }

        return nodeType;
    }
});
