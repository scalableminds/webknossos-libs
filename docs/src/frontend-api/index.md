# Frontend Scripting API

WEBKNOSSOS exposes a JavaScript API in the browser for user scripts, e.g. to add custom keyboard shortcuts or to automate repetitive annotation steps.
Scripts can be run from the annotation viewer or attached to tasks, see [Scripts](../webknossos/tasks_projects/scripts.md).

The API is available as `window.webknossos` once an annotation or dataset is opened.
Request the API with `apiReady`, which resolves as soon as WEBKNOSSOS is fully initialized:

```javascript
window.webknossos.apiReady(3).then((api) => {
  const nodeId = api.tracing.getActiveNodeId();
  api.tracing.setCommentForNode("my comment", nodeId);
});
```

The resolved `api` object is split into the following namespaces:

| Namespace | Description |
|---|---|
| [`api.tracing`](tracing.md) | Skeleton and volume annotation: nodes, trees, segments, groups, camera and tools. |
| [`api.data`](data.md) | Layers, mappings, meshes, bounding boxes, data and screenshots. |
| [`api.user`](user.md) | User configuration. |
| [`api.utils`](utils.md) | Utilities such as keyboard shortcuts, toasts and event listeners. |

## `window.webknossos`

::: api_loader::default
    handler: typescript
