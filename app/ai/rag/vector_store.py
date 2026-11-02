
import asyncio
from pgvector import Vector

from app.utils.ids import generate_uuid
from app.DB.pgvector import get_connection


class VectorStore:

    async def get_document_by_hash(self, file_hash: str) -> dict | None:
        def _get():
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT document_id, filename, file_hash,
                               file_size_bytes, uploaded_at, ingested_at,
                               ingestion_status, chunk_count,
                               chunking_version, embedding_model
                        FROM public.documents
                        WHERE file_hash = %s
                        LIMIT 1
                        """,
                        (file_hash,)
                    )

                    row = cur.fetchone()

                    if not row:
                        return None

                    columns = [desc.name for desc in cur.description]
                    return dict(zip(columns, row))

        return await asyncio.to_thread(_get)

    async def get_document_by_id(self, document_id: str) -> dict | None:
        def _get():
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT document_id, filename, file_hash,
                               file_size_bytes, uploaded_at, ingested_at,
                               ingestion_status, chunk_count,
                               chunking_version, embedding_model
                        FROM public.documents
                        WHERE document_id = %s
                        LIMIT 1
                        """,
                        (document_id,)
                    )

                    row = cur.fetchone()

                    if not row:
                        return None

                    columns = [desc.name for desc in cur.description]
                    return dict(zip(columns, row))

        return await asyncio.to_thread(_get)

    async def insert(
        self,
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings count mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        def _insert():
            with get_connection() as conn:
                with conn.cursor() as cur:

                    # 1. Insert document metadata
                    cur.execute(
                        """
                        INSERT INTO public.documents (
                            document_id,
                            filename,
                            file_hash,
                            file_size_bytes,
                            uploaded_at,
                            ingested_at,
                            ingestion_status,
                            chunk_count,
                            chunking_version,
                            embedding_model
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            document_record["document_id"],
                            document_record["filename"],
                            document_record["file_hash"],
                            int(document_record["file_size_bytes"]),
                            str(document_record["uploaded_at"]),
                            str(document_record["ingested_at"]),
                            str(document_record["ingestion_status"]),
                            int(document_record["chunk_count"]),
                            str(document_record["chunking_version"]),
                            str(document_record["embedding_model"]),
                        )
                    )

                    # 2. Insert document chunks and embeddings
                    for chunk, embedding in zip(chunks, embeddings):
                        chunk_id = chunk.get("id") or generate_uuid()

                        cur.execute(
                            """
                            INSERT INTO public.chunks (
                                id,
                                document_id,
                                filename,
                                chunk_index,
                                page_number,
                                text,
                                embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                chunk_id,
                                document_record["document_id"],
                                document_record["filename"],
                                int(chunk["chunk_index"]),
                                int(chunk["page_number"]),
                                chunk["text"],
                                Vector(embedding),
                            )
                        )

        await asyncio.to_thread(_insert)

    async def retrieve(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:

        def _retrieve():
            with get_connection() as conn:
                with conn.cursor() as cur:

                    sql = """
                        SELECT
                            id,
                            text,
                            document_id,
                            filename,
                            chunk_index,
                            page_number,
                            embedding <=> %s AS distance
                        FROM public.chunks
                    """

                    params = [Vector(query_embedding)]

                    if document_id:
                        sql += " WHERE document_id = %s"
                        params.append(document_id)

                    sql += """
                        ORDER BY embedding <=> %s
                        LIMIT %s
                    """

                    # The ordering expression needs the query embedding again.
                    params.extend([Vector(query_embedding), n_results])

                    cur.execute(sql, params)
                    rows = cur.fetchall()

                    return [
                        {
                            "id": row[0],
                            "document": row[1],
                            "metadata": {
                                "document_id": row[2],
                                "filename": row[3],
                                "chunk_index": row[4],
                                "page_number": row[5],
                            },
                            "distance": float(row[6]),
                        }
                        for row in rows
                    ]

        return await asyncio.to_thread(_retrieve)


vector_store = VectorStore()
