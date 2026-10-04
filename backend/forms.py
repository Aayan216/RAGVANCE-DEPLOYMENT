from django import forms
from django.conf import settings
from backend.models import Document


class DocumentUploadForm(forms.ModelForm):
    def __init__(self, *args, document_count=None, **kwargs):
        # document_count: number of EXISTING Document records, passed by the
        # view so the limit is enforced from real DB state only (chunks,
        # FAISS vectors, browser state and failed uploads are never counted).
        super().__init__(*args, **kwargs)
        self.document_count = document_count

    class Meta:
        model = Document
        fields = ["title", "file"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "Document title (optional)"}),
            "file": forms.FileInput(attrs={"class": "form-control", "accept": ".pdf,.docx,.pptx,.txt"}),
        }

    def clean_file(self):
        # Count check first: it is raised as a field error so the existing
        # template's form.file.errors block always renders the message.
        max_docs = int(getattr(settings, "MAX_DOCUMENTS", 20))
        if self.document_count is not None and self.document_count >= max_docs:
            raise forms.ValidationError(
                f"Maximum of {max_docs} documents allowed."
            )

        file = self.cleaned_data["file"]
        allowed_types = ["pdf", "docx", "pptx", "txt"]
        ext = file.name.split(".")[-1].lower()
        if ext not in allowed_types:
            raise forms.ValidationError("Unsupported file type. Allowed: PDF, DOCX, PPTX, TXT")

        max_mb = int(getattr(settings, "MAX_UPLOAD_FILE_SIZE_MB", 25))
        max_bytes = max_mb * 1024 * 1024
        if file.size > max_bytes:
            raise forms.ValidationError(
                f"File is too large. Maximum allowed size is {max_mb} MB."
            )
        return file


class TutorQuestionForm(forms.Form):
    question = forms.CharField(
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 3,
            "placeholder": "Ask a question about your study materials..."
        }),
        label=""
    )