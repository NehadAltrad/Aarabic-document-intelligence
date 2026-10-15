"""Streamlit user interface for Arabic Document Intelligence.

Start: streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from docintel.classification import predict_category, train_classifier
from docintel.documents import DocumentError, make_chunks, parse_document
from docintel.indicators import extract_indicators
from docintel.retrieval import DocumentRetriever

MAX_DOCUMENTS = 8

st.set_page_config(
    page_title="Arabic Document Intelligence",
    page_icon="📄",
    layout="wide",
)
st.markdown(
    """
<style>
    .stApp { direction: rtl; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { text-align: right; }
    [data-testid="stMetric"] { direction: rtl; }
    .app-subtitle { color: #54717b; margin-bottom: 1.4rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="تجهيز نموذج التصنيف...")
def get_classifier():
    # Public synthetic examples only; no uploaded document data is in the shared model cache.
    return train_classifier()


st.title("📄 Arabic Document Intelligence")
st.markdown(
    '<p class="app-subtitle">مساعد تجريبي لفهم المستندات العربية: استخراج، تصنيف، وأسئلة مدعومة بالدليل.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("عن المشروع")
    st.markdown("**التقنيات:** Python · Scikit-learn · TF-IDF · Streamlit · PyMuPDF")
    st.info("الردود تعرض مقاطع من المصدر ولا تُنشئ إجابات جديدة. يجب مراجعة الدليل يدويًا.")
    st.warning("استخدمي ملفات تجريبية أو عامة فقط. لا ترفعي مستندات سرية أو بيانات شخصية.")
    st.caption("الحدود: 8 ملفات، 5 MB لكل ملف، و30 صفحة لكل PDF. ملفات PDF المصوّرة تحتاج OCR غير متوفر هنا.")

uploaded_files = st.file_uploader(
    "ارفعي مستندات عربية نصية",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="يمكنك استخدام الملفات التجريبية داخل data/samples في المستودع.",
)
if not uploaded_files:
    st.info("ابدئي برفع ملف أو أكثر بصيغة PDF أو TXT. تتوفر أمثلة في مجلد data/samples داخل المشروع.")
    st.stop()
if len(uploaded_files) > MAX_DOCUMENTS:
    st.error(f"يمكن معالجة {MAX_DOCUMENTS} ملفات كحد أقصى في المرة الواحدة.")
    st.stop()

pages_by_source = {}
all_pages = []
for file_number, uploaded_file in enumerate(uploaded_files, start=1):
    source_name = f"{file_number}. {uploaded_file.name}"
    try:
        pages = parse_document(source_name, uploaded_file.getvalue())
    except DocumentError as error:
        st.error(f"تعذرت معالجة {uploaded_file.name}: {error}")
        continue
    pages_by_source[source_name] = pages
    all_pages.extend(pages)

if not pages_by_source:
    st.stop()

classifier = get_classifier()
chunks = make_chunks(all_pages)
retriever = DocumentRetriever(chunks)
predictions = {
    source: predict_category(classifier, "\n".join(page.text for page in pages))
    for source, pages in pages_by_source.items()
}

a, b, c = st.columns(3)
a.metric("المستندات", len(pages_by_source))
b.metric("المقاطع المفهرسة", len(chunks))
c.metric("الصفحات النصية", len(all_pages))

tab_classify, tab_question, tab_overview = st.tabs(
    ["🏷️ تحليل وتصنيف", "🔍 اسأل المستندات", "📊 ملخص وتصدير"]
)
with tab_classify:
    selected_source = st.selectbox("المستند", list(pages_by_source))
    selected_pages = pages_by_source[selected_source]
    full_text = "\n".join(page.text for page in selected_pages)
    guess = predictions[selected_source]
    st.subheader(f"التصنيف المقترح: {guess.category}")
    st.caption(
        f"درجة ترجيح النموذج: {guess.score:.1%} (احتمال غير معاير وليس ضمانًا للدقة)."
    )
    st.markdown("#### معلومات مستخرجة بقواعد بسيطة")
    indicators = extract_indicators(full_text)
    for title, values in indicators.items():
        st.write(f"**{title}:** " + ("، ".join(values[:20]) if values else "لم يُرصد"))
    with st.expander("عرض النص المستخرج"):
        st.text(full_text[:12_000])

with tab_question:
    question = st.text_input(
        "ما سؤالك عن المستندات المرفوعة؟",
        placeholder="مثال: ما المطلوب قبل مشاركة السجلات الحساسة؟",
        key="question_text",
    )
    if st.button("البحث عن دليل", type="primary", use_container_width=True):
        answer = retriever.ask(question, top_k=3)
        if not answer.found:
            st.warning(answer.message)
        else:
            st.success(answer.message)
            for number, item in enumerate(answer.evidence, start=1):
                with st.container(border=True):
                    page_label = f"الصفحة {item.page}" if item.page is not None else "ملف نصي"
                    st.markdown(f"**الدليل {number}:** {item.source} · {page_label}")
                    st.caption(f"تشابه نصي: {item.similarity:.1%} — ليس مقياس ثقة في صحة الإجابة")
                    st.write(item.excerpt)

with tab_overview:
    summary_rows = [
        {
            "المستند": source,
            "التصنيف المقترح": item.category,
            "درجة الترجيح": round(item.score, 3),
            "عدد الصفحات النصية": len(pages_by_source[source]),
        }
        for source, item in predictions.items()
    ]
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)
    export_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Automated baseline predictions only. Review before making decisions.",
        "documents": summary_rows,
    }
    st.download_button(
        "تحميل ملخص التصنيف (JSON)",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="arabic-document-analysis.json",
        mime="application/json",
    )
    st.caption("لا يتضمن الملخص النصوص الأصلية. يجري التحليل في ذاكرة جلسة التطبيق دون حفظ ملفات المستخدم في مجلد المشروع.")
