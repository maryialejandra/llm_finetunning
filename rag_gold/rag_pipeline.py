from typing import Any, List
from collections import Counter

import pandas as pd
import faiss
import re
from tqdm import tqdm

from llama_index.core import (
    Document,
    QueryBundle
)

# from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.retrievers import BaseRetriever
# from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.groq import Groq


from src.utils import get_secret

# The following loads file with environment variables (such as LLM API Keys)
# change path to file as needed

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DEFAULT_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Set llm
GROQ_MODEL = "llama-3.1-70b-versatile"


def make_query_engine(documents: list[Document],
                      chunk_size: int = 288,
                      chunk_overlap: int = 100,
                      similarity_top_k: int = 5,
                      query_mode: str = "default",
                      embed_model: HuggingFaceEmbedding | None = None,
                      llm = None,
                      vs_type: str = "simple"
                      ) -> RetrieverQueryEngine:

    llm = llm or Groq(GROQ_MODEL, api_key=get_secret("GROQ_API_KEY"), temperature=0.1 )
    print(f"Using LLM={llm}")
    embed_model = embed_model or HuggingFaceEmbedding(DEFAULT_EMB_MODEL)

    if vs_type == "simple":
        vecstore_idx = make_vectore_index(embedding=embed_model,
                                      documents=documents,
                                      chunk_size=chunk_size,
                                      chunk_overlap=chunk_overlap
                                      )
        vector_store = vecstore_idx.vector_store
        id_2_node = vecstore_idx.docstore.docs
    else:
        id_2_node, vector_store = populate_vecstore_faiss(
                                        documents,
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap,
                                        embed_model=embed_model
                                        )
    retriever = VectorDBRetriever(
        vector_store,
        id_2_node,
        embed_model,
        query_mode=query_mode,
        similarity_top_k=similarity_top_k,
        verbose=True
    )

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    return query_engine



def make_vectore_index(embedding: HuggingFaceEmbedding,
                       documents: list[Document],
                       chunk_size: int,
                       chunk_overlap: int
                       ) -> VectorStoreIndex:
    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            embedding
        ]
    )

# run the pipeline
    nodes = pipeline.run(documents=documents)

    return VectorStoreIndex(nodes, show_progress=True)


def populate_vecstore_faiss(
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
        embed_model: HuggingFaceEmbedding
    ) -> tuple[dict[str, TextNode], BasePydanticVectorStore]:

    sample_emb = embed_model.get_text_embedding("Hello worlds!")
    embed_dim = len(sample_emb)
    print(f"Embedding dimension: {embed_dim}")

    text_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    nodes = make_text_nodes(documents, text_parser)
    compute_embeddings_ip(nodes, embed_model)

    id_2_node, vec_store = make_vec_store_faiss(nodes, embed_dim)

    return id_2_node, vec_store


def make_text_nodes(documents, text_parser) -> list[TextNode]:
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    print("chunks:",  len(text_chunks))
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    print("nodes:", len(nodes))

    return nodes

def compute_embeddings_ip(nodes: list[TextNode],
                          embed_model: HuggingFaceEmbedding) -> None:
    print("computing embeddings now...")
    for node in tqdm(nodes):
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding


def make_vec_store_faiss(nodes: list[TextNode], embed_dim: int):

    faiss_index = faiss.IndexFlatL2(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # vector_store = SimpleVectorStore()
    print(f"adding {len(nodes)} to vector_store")
    ids = vector_store.add(nodes)
    id_2_node = {id_: node for (id_, node) in zip(ids, nodes)}

    return id_2_node, vector_store


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        id_2_node: dict[str, TextNode],
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
        verbose: bool = False
    ) -> None:
        """Init params."""
        super().__init__()
        self._vector_store = vector_store
        self._id_2_node = id_2_node
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self._verbose = verbose


    def _debug_query(self, query: str):
        query_embedding = self._embed_model.get_query_embedding(
            query
        )

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )

        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for idx, node_id in enumerate(query_result.ids):
            node = self._id_2_node[node_id]
            score = query_result.similarities[idx]
            if self._verbose:
                print(f"\n==== idx: {idx}  node_id: {node_id}  SCORE: {score:.4f}====\n{node.get_text()}\n")
            nodes_with_scores.append({"node_id": node_id,
                                      "node_text": node.text,
                                      "score": score})

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for idx, node_id in enumerate(query_result.ids):
            node = self._id_2_node[node_id]
            score = query_result.similarities[idx]
            if self._verbose:
                print(f"\n==== idx: {idx}  node_id: {node_id}====\n{node.get_text()}\n")
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

def extract_letter(ans: str, verbose=False) -> str:
    m = re.match(r"([a-dA-D])(\)|\.)*", ans)
    if m:
        return m.group(1)
    else:
        if verbose:
            print("???", ans)
        return "?"


def combine_answers_models(ans_vers):

    ret_df = None
    for model_key, file_path in ans_vers.items():
        df = pd.read_csv(file_path)
        assert df.columns[0] == 'ID'
        df.columns = ['ID', model_key]
        df[model_key] = df[model_key].apply(extract_letter)
        if ret_df is None:
            ret_df = df
        else:
            ret_df = pd.merge(ret_df, df, on='ID')

        del df

    ans_cols = [col for col in ret_df.columns if col != 'ID']
    print(ans_cols)

    def all_ans(row) -> str:
        # print(row)
        anss = [ row[col] for col in ans_cols ]
        if len(set(anss)) == 1:
            return anss[0]
        else:
            return '?'


    def cnts(row) -> str:
        return dict(Counter(row[ans_cols]))

    def ans_agree_2(row) -> str:
        """The ones that answered agree answer and they agree rest don't know"""
        # print(row)

        d = row['cnts'].copy()
        if '?' in d:
            del d['?']

        if len(d) == 1 and next(iter(d.values())) >= 2:
            return next(iter(d.keys()))
        else:
            return "?"


    def ans_agree_1(row) -> str:
        """The ones that answered agree answer and they agree rest don't know"""
        # print(row)

        d = row['cnts'].copy()
        if '?' in d:
            del d['?']

        if len(d) == 1:
            return next(iter(d.keys()))
        else:
            return "?"

    ret_df['cnts'] = ret_df.apply(cnts, axis=1)
    ret_df['all'] = ret_df.apply(all_ans, axis=1)
    ret_df['ans_agree_1'] = ret_df.apply(ans_agree_1, axis=1)
    ret_df['ans_agree_2'] = ret_df.apply(ans_agree_2, axis=1)

    return ret_df
