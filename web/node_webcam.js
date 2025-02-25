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

        const refresh_cameras = async(widget, force=false) => {
            let url = "/jov_capture/camera"
            if (force) {
                url += "?force=true";
            }
            var data = await api_get(url);
            widget.options.values = data;
            widget.value = data[0];
            app.canvas.setDirty(true);
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_camera = this.widgets.find(w => w.name == 'CAMERA');

            this.addWidget('button', 'REFRESH CAMERA LIST', 'refresh', async () => {
                await refresh_cameras(widget_camera);
            });
            await refresh_cameras(widget_camera, true);
            return me;
        }

        return nodeType;
    }
});
