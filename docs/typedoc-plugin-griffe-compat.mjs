// @ts-check
// TypeDoc plugin that keeps TypeDoc's JSON output loadable by griffe-typedoc
// 0.1.1 (used by mkdocstrings-typescript):
// - Types are serialized as their plain string representation, e.g.
//   `Promise<ApiInterface>`. griffe-typedoc can't decode several type kinds
//   (e.g. "unknown" or "indexedAccess") and the signature template in
//   templates/ only needs the string anyway.
// - The top-level `schemaVersion` field is dropped (fixed upstream but not
//   yet released).

import { Serializer, Type } from "typedoc";

/** @param {import("typedoc").Application} app */
export function load(app) {
  app.serializer.addSerializer({
    priority: -1,
    supports: (item) => item instanceof Type,
    toObject: (item) => ({ type: "intrinsic", name: item.toString() }),
  });
  app.serializer.on(Serializer.EVENT_END, (event) => {
    if (event.output) {
      delete event.output.schemaVersion;
    }
  });
}
