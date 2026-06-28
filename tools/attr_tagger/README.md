# Attribute Tagger

A tiny web tool to add **per-instance attributes** to a COCO dataset (detection
or segmentation). The instances already exist — you just click each object and
pick its attribute values. One file, no dependencies (Python stdlib only).

The output is plain COCO + an attribute schema, and trains directly with this
repo (the loader reads the attribute heads automatically).

## Try it

```bash
python3 tools/attr_tagger/tag.py \
    --coco   tools/attr_tagger/sample/coco.json \
    --images tools/attr_tagger/sample/images
```

Open <http://127.0.0.1:7860>. The included sample (8 COCO images, no attributes
yet) starts on the **define** screen so you can see the whole flow.

## How it works

1. **Define attributes** — if the dataset has no attribute schema, the first
   screen lets you add attributes (a name + comma-separated options, e.g.
   `color = red, green, blue`). Done once.
2. **Tag** — go image by image, click an object, click its attribute values.
   Objects are colored: blue = untagged, green ✓ = tagged, yellow = selected.
3. **Resume** — re-run and it jumps to the first image that still has an
   untagged object. If attributes were never defined, you get the define screen.

Saved automatically (and on exit). The input file is never modified: results go
to `<coco_stem>.tagged.json` next to it, and that file is loaded back on resume.

### Keys

| key | action |
|-----|--------|
| click | select an object |
| `1`–`9` / `` ` `` | set / clear the first attribute on the selected object |
| `←` `→` | previous / next image |
| `n` | jump to next image with untagged objects |
| `s` | save now · `Esc` deselect |

## Options

| flag | default | meaning |
|------|---------|---------|
| `--coco` | — | input COCO JSON (required) |
| `--images` | — | directory of images (required) |
| `--out` | `<coco_stem>.tagged.json` | output path (resumed if it exists) |
| `--port` | `7860` | server port |

## Output format

Standard COCO plus a top-level `attributes` schema and a per-annotation
`attributes` map:

```jsonc
"attributes": [
  { "name": "color", "categories": [{ "id": 0, "name": "red" }, { "id": 1, "name": "green" }] }
],
"annotations": [
  { "id": 1, "image_id": 1, "category_id": 7, "bbox": [...], "segmentation": [...],
    "attributes": { "color": 1 } }
]
```

Untagged instances simply omit the value; the training loader treats those as
ignored (`-100`), so a partially-tagged file is still valid to train on.
