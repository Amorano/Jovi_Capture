/**
 * File: node_window.js
 * Project: jov_capture
 */

import { app } from "../../../scripts/app.js";
import { api_get } from './util_jov.js'

const _id = "WINDOW (JOV_CAPTURE)";

app.registerExtension({
	name: 'jov_capture.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);

            const widget_window = this.widgets.find(w => w.name == 'WINDOW');

            this.addWidget('button', 'REFRESH WINDOW LIST', 'refresh', async () => {
                var data = await api_get("/jov_capture/window");
                widget_window.options.values = Object.keys(data);
                widget_window.value = widget_window.options.values[0];
                console.info(widget_window)
                app.canvas.setDirty(true);
            });
            return me;
        }

        return nodeType;
    }
});
