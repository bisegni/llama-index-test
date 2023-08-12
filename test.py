from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    chunk_size=512,
    )

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
    )
index.storage_context.persist()
query_engine = index.as_query_engine(
    service_context=service_context,
    streaming=True,
    similarity_top_k=1
    )
response = query_engine.query("how natilus can be comapred to moder submarine?")
for text in response.response_gen:
    print(text, end="")
