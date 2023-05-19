# Marqo

[Marqo](https://www.pinecone.io) is an end-to-end, multimodal vector search engine. With Marqo, users can store and query unstructured data such as text, images, and code through a single easy-to-use API. Input preprocessing, machine learning inference, and storage are all included out of the box and can be easily scaled. You can self host Marqo [with out opensource docker image](https://github.com/marqo-ai/marqo#getting-started) or you can [sign up to our cloud for a managed solution](https://www.marqo.ai/pricing).

A full Jupyter notebook walkthrough for the Marqo flavor of the retrieval plugin can be found [here](https://github.com/openai/chatgpt-retrieval-plugin/blob/main/examples/providers/pinecone/semantic-search.ipynb).

The app will create a Marqo index for you automatically if you are not planning on using an existing one, just set the corresponding environment variable.

**Environment Variables:**

| Name                                | Required | Description                                                                                                                           |
| ----------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `DATASTORE`                         | Yes      | Datastore name, set this to `pinecone`                                                                                                |
| `BEARER_TOKEN`                      | Yes      | Your secret token for authenticating requests to the API                                                                              |
| `OPENAI_API_KEY`                    | No       | Marqo generates embeddings for you so this is not required                                                                            |
| `MARQO_API_URL`                     | Yes      | Your Pinecone API key, found in the [Pinecone console](https://app.pinecone.io/)                                                      |
| `MARQO_API_KEY`                     | Maybe    | Your API key for Marqo, this is only required if you are using the managed cloud offering, keys are found in your Marqo console.      |
| `MARQO_INDEX`                       | Yes      | Your chosen Marqo index name. **Note:** Index name must consist of lower case alphanumeric characters or '-'                          |
| `MARQO_INFERENCE_MODEL`             | No       | Your chosen Model for generating embeddings, [see our documentation](https://docs.marqo.ai/0.0.20/Models-Reference/dense_retrieval/)|
| `MARQO_UPSERT_BATCH_SIZE`           | No       | Batch size for bulk upserts                                                                                                           |
| `TREAT_URLS_AND_POINTERS_AS_IMAGES` | No       | `True` or `False`, if `True` then images will be used to downloaded and embedded. Require that `MARQO_INFERENCE_MODEL` is a CLIP model|

If you want more control over index creation or want to load your own custom inference models then this can be done through the Marqo API, [please refer to the documentation](https://docs.marqo.ai/latest/).

```python
# Creating index with Marqo's SDK - use only if you wish to create the index manually.

import os
import marqo

mq = marqo.Client(url=os.environ['MARQO_API_URL'], api_key=os.environ['MARQO_API_KEY'])

settings = {
    "index_defaults": {
        "treat_urls_and_pointers_as_images": False,
        "model": "hf/all_datasets_v4_MiniLM-L6",
        "normalize_embeddings": True,
        "text_preprocessing": {
            "split_length": 2,
            "split_overlap": 0,
            "split_method": "sentence"
        },
        "image_preprocessing": {
            "patch_method": None
        },
        "ann_parameters" : {
            "space_type": "cosinesimil",
            "parameters": {
                "ef_construction": 128,
                "m": 16
            }
        }
    },
    "number_of_shards": 5,
    "number_of_replicas": 1
}

mq.create_index(os.environ['MARQO_INDEX'], settings_dict=settings)
```
