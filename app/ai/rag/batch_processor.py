
import logging
from app.ai.rag.embed import embedding

logger = logging.getLogger("uvicorn.error")


class BatchProcessor:

    def __init__(self, batch_size: int = 10):
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero")

        self.batch_size = batch_size

    async def process_batches(self, chunks: list[str]) -> list[list[float]]:

        if not chunks:
            logger.info("[BATCH] No chunks to process.")
            return []

        all_embeddings = []
        total_chunks = len(chunks)

        total_batches = (
            total_chunks + self.batch_size - 1
        ) // self.batch_size

        logger.info("========== BATCH PROCESSING STARTED ==========")
        logger.info(f"[BATCH] Total chunks: {total_chunks}")
        logger.info(f"[BATCH] Batch size: {self.batch_size}")
        logger.info(f"[BATCH] Total batches: {total_batches}")

        for start in range(0, total_chunks, self.batch_size):

            batch_number = start // self.batch_size + 1
            batch = chunks[start:start + self.batch_size]

            logger.info(
                f"[BATCH] Processing batch {batch_number}/{total_batches} "
                f"with {len(batch)} chunks"
            )

            batch_embeddings = await embedding.embed_chunks(batch)

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    f"Batch {batch_number}: expected {len(batch)} embeddings, "
                    f"got {len(batch_embeddings)}"
                )

            all_embeddings.extend(batch_embeddings)

            logger.info(
                f"[BATCH] Batch {batch_number} completed successfully"
            )

        logger.info(
            f"[BATCH] Total embeddings generated: {len(all_embeddings)}"
        )

        logger.info("========== BATCH PROCESSING COMPLETED ==========")

        return all_embeddings


batch_processor = BatchProcessor(batch_size=10)
