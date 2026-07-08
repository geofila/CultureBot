# `dataset/` — your RAG / Knowledge-Graph dataset goes here

The real dataset is **confidential and is not included** in this repository. Everything
in this folder is gitignored except this README and the `*.example.*` files, which
document the exact formats the pipeline expects.

To run the bot with your own data, drop the following files into this folder
(filenames must match exactly — they are what `compose.yml` mounts into the
`pipelines` container), then **uncomment the `volumes:` block** of the `pipelines`
service in `compose.yml`