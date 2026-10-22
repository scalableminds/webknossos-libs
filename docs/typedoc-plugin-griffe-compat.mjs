// @ts-check
// TypeDoc plugin that keeps TypeDoc's JSON output loadable by griffe-typedoc
// (used by mkdocstrings-typescript). griffe-typedoc only knows a subset of
// TypeDoc's JSON output and fails on everything else:
// - Types are serialized as their plain string representation, e.g.
//   `Promise<ApiInterface>`. griffe-typedoc can't decode several type kinds
//   (e.g. "unknown" or "indexedAccess") and the signature template in
//   templates/ only needs the string anyway.
// - Unsupported block tags (e.g. JSDoc-only tags such as `@memberof` or
//   `@version`) are dropped.
// - The top-level `schemaVersion` field is dropped (not accepted by
//   griffe-typedoc 0.1.1, fixed upstream but not yet released).

import { Comment, Serializer, Type } from "typedoc";

// Keep in sync with griffe_typedoc's BlockTagKind enum.
const SUPPORTED_BLOCK_TAGS = new Set([
  "@alpha",
  "@beta",
  "@category",
  "@defaultValue",
  "@deprecated",
  "@enum",
  "@event",
  "@eventProperty",
  "@example",
  "@experimental",
  "@group",
  "@hidden",
  "@ignore",
  "@inheritDoc",
  "@interface",
  "@internal",
  "@label",
  "@link",
  "@module",
  "@namespace",
  "@overload",
  "@override",
  "@packageDocumentation",
  "@param",
  "@private",
  "@privateRemarks",
  "@property",
  "@protected",
  "@public",
  "@readonly",
  "@remarks",
  "@returns",
  "@satisfies",
  "@sealed",
  "@see",
  "@template",
  "@throws",
  "@typeParam",
  "@virtual",
]);

/** @param {import("typedoc").Application} app */
export function load(app) {
  app.serializer.addSerializer({
    priority: -1,
    supports: (item) => item instanceof Type,
    toObject: (item) => ({ type: "intrinsic", name: item.toString() }),
  });
  app.serializer.addSerializer({
    priority: -1,
    supports: (item) => item instanceof Comment,
    toObject: (_item, obj) => ({
      ...obj,
      blockTags: obj.blockTags?.filter((tag) => SUPPORTED_BLOCK_TAGS.has(tag.tag)),
    }),
  });
  app.serializer.on(Serializer.EVENT_END, (event) => {
    if (event.output) {
      delete event.output.schemaVersion;
    }
  });
}
