/**
 * File: util_jov.js
 * Project: Jovi_Capture
 *
 */

import { api } from "../../scripts/api.js"

export async function api_get(route) {
    var response = await api.fetchApi(route, { cache: "no-store" });
    var text = await response.text();
    return JSON.parse(text);
}
