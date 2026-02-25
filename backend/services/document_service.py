"""Document management service stub."""
import logging
logger = logging.getLogger(__name__)

class DocumentService:
    def list_documents(self):
        return []
    def get_document(self, doc_id: str):
        return None

document_service = DocumentService()
