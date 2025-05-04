/**
 * File: node_window.js
 * Project: jov_capture
 */

import { app } from "../../scripts/app.js";
import { api_get } from './util_jov.js'

const _id = "WINDOW (JOV_CAPTURE)";

app.registerExtension({
	name: 'jov_capture.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const refresh_windows = async(widget) => {
            var data = await api_get("/jov_capture/window");
            widget.options.values = Object.keys(data);
            widget.value = widget.options.values[0];
            app.canvas.setDirty(true);
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_window = this.widgets.find(w => w.name == 'window');

            this.addWidget('button', 'REFRESH WINDOW LIST', 'refresh', async () => {
                refresh_windows(widget_window);
            });
            await refresh_windows(widget_window);
            return me;
        }

        return nodeType;
    }
});
