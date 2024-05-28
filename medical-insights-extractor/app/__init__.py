from app.manage import create_app

app, logger = create_app()

from app.resource.index import Index
from app.resource.medical_document_insight_extraction.medical_insight_extraction import MedicalInsightsExtractor, QnAExtractor

app.router.add_view('/', Index)
app.router.add_route('POST', '/medical-insights/qna-extract', QnAExtractor.post)
app.router.add_route('POST', '/medical-insights/analyze', MedicalInsightsExtractor.post)
app.router.add_route('GET', '/medical-insights/get/extract', MedicalInsightsExtractor.get)