import re
import textwrap
from typing import List, NamedTuple

from danswer.configs.app_configs import BLURB_SIZE, MINI_CHUNK_SIZE, SKIP_METADATA_IN_CHUNK
from danswer.configs.model_configs import DOC_EMBEDDING_CONTEXT_SIZE
from danswer.connectors.models import Document
from danswer.indexing.chunker import Chunker, _get_metadata_suffix_for_document_index
from danswer.indexing.indexing_heartbeat import Heartbeat
from danswer.indexing.models import DocAwareChunk
from danswer.llm.factory import get_default_llms
from danswer.natural_language_processing.utils import BaseTokenizer
from danswer.utils.logger import setup_logger

logger = setup_logger()

CHUNK_OVERLAP = 0
MAX_METADATA_PERCENTAGE = 0.25
CHUNK_MIN_CONTENT = 256


class Section(NamedTuple):
    start_line: int
    end_line: int
    title: str
    summary: str


class RealLLM:
    def __init__(self):
        self.llm = get_default_llms()[0]

    def analyze_document(self, document_text: str) -> tuple[str, str]:
        prompt = textwrap.dedent(
            f"""
            Given the following document text, provide a concise title and summary (1-2 sentences) in the language of the document.
            Format: Title|||Summary

            Example:
            Обзор Проекта|||Документ описывает текущий статус и планы развития проекта.

            Result should include exactly one separator |||.

            Document text:
            {document_text}
        """
        ).strip()

        result = self.llm._invoke_implementation(prompt=prompt).content

        try:
            title, summary = result.split("|||")
            return title.strip(), summary.strip()
        except ValueError as e:
            print(f"Error parsing document analysis result: {e}")
            return "Документ", "Общий обзор документа"

    def analyze_sections(self, document_text: str) -> list[Section]:
        prompt = textwrap.dedent(
            f"""
            Given the following document text with line numbers, identify main semantic sections.
            Sections can vary in length, but should generally be anywhere from a few paragraphs to a few pages long.

            For each section, provide:
            1. Starting line number (just the number, e.g. 5 not [5])
            2. Ending line number (just the number)
            3. Section title STRICTLY in the same langauge as the main part of the document
            4. Brief section summary STRICTLY in the same langauge as the main part of the document (1-2 sentences)

            Format each section exactly like this:
            1|||11|||Общее положение|||Описание общей ситуации и целей
            12|||25|||ММГ|||Обсуждение проекта ММГ и его статуса

            IMPORTANT: 
            - Look for clear section headers in the text and use them as titles whenever relevant
            - If the document looks like chat, you can group related messages together
            - Make sure line numbers correspond to actual text lines
            - Include ALL text in sections, don't leave any text uncovered
            - Don't overlap sections
            - Use EXACTLY the format shown in example above
            - Keep document and section summaries concise (2-3 sentences max)
            - Group related content together (questions with their topics)

            Document text with line numbers:
            {document_text}
        """
        ).strip()

        result = self.llm._invoke_implementation(prompt=prompt).content

        sections = []
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            try:
                parts = line.split("|||")
                if len(parts) != 4:
                    print(f"Warning: Invalid section format: {line}")
                    continue

                start, end, title, summary = parts

                # Extract numbers from potential [X] format
                start_match = re.search(r"\d+", start.strip())
                end_match = re.search(r"\d+", end.strip())

                if not start_match or not end_match:
                    print(f"Warning: Could not extract line numbers from: {line}")
                    continue

                sections.append(
                    Section(
                        start_line=int(start_match.group()),
                        end_line=int(end_match.group()),
                        title=title.strip(),
                        summary=summary.strip(),
                    )
                )
            except Exception as e:
                print(f"Error processing section line '{line}': {e}")
                continue

        return sections


class MockLLM:
    """Mock LLM for testing purposes"""

    def __init__(self):
        pass

    def analyze_document(self, document_text: str, language: str | None = None) -> tuple[str, str]:
        """Mock document analysis returning title and summary"""
        return "Mock Document Title", "This is a mock document summary."

    def analyze_sections(self, document_text: str, language: str | None = None) -> List[Section]:
        """Mock section analysis returning predefined sections"""
        lines = document_text.split("\n")
        if not lines:
            return []

        # For testing, create one section for every 10 lines
        sections = []
        current_line = 0
        while current_line < len(lines):
            end_line = min(current_line + 10, len(lines))
            sections.append(
                Section(
                    start_line=current_line,
                    end_line=end_line,
                    title=f"Section {len(sections) + 1}",
                    summary=f"This is section {len(sections) + 1} summary.",
                )
            )
            current_line = end_line
        return sections


class ContextChunker(Chunker):
    """Chunks documents into semantically cohesive sections with doc and section summaries using LLM"""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        enable_multipass: bool = False,
        enable_large_chunks: bool = False,
        blurb_size: int = BLURB_SIZE,
        include_metadata: bool = not SKIP_METADATA_IN_CHUNK,
        chunk_token_limit: int = DOC_EMBEDDING_CONTEXT_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        mini_chunk_size: int = MINI_CHUNK_SIZE,
        heartbeat: Heartbeat | None = None,
        llm: RealLLM | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            enable_multipass=enable_multipass,
            enable_large_chunks=enable_large_chunks,
            blurb_size=blurb_size,
            include_metadata=include_metadata,
            chunk_token_limit=chunk_token_limit,
            chunk_overlap=chunk_overlap,
            mini_chunk_size=mini_chunk_size,
            heartbeat=heartbeat,
        )

        self.llm = llm or MockLLM()

    def _add_line_numbers(self, text: str) -> str:
        """Add line numbers to each line of text"""
        lines = text.split("\n")
        return "\n".join(f"[{i+1}] {line}" for i, line in enumerate(lines))

    def _extract_lines(self, text: str, start: int, end: int) -> str:
        """Extract lines from text between start and end line numbers"""
        lines = text.split("\n")
        return "\n".join(lines[start - 1 : end])

    def _get_document_context(self, document: Document, language: str | None = None) -> str:
        """Generate document context using LLM"""
        document_text = "\n".join(section.text for section in document.sections)
        title, summary = self.llm.analyze_document(document_text, language)
        return f"Document context: this excerpt is from a document titled '{title}'. {summary}"

    def _get_section_context(self, section: Section) -> str:
        """Generate section context"""
        return f"Section context: this excerpt is from the section titled '{section.title}'. {section.summary}"

    def _chunk_section(
        self,
        document: Document,
        section_text: str,
        section_context: str,
        document_context: str,
        content_token_limit: int,
    ) -> list[DocAwareChunk]:
        """Chunk a single section with its context"""
        chunks = []
        split_texts = self.chunk_splitter.split_text(section_text)

        for i, split_text in enumerate(split_texts):
            chunk_text = f"{document_context}\n{section_context}\n{split_text}"

            # Create chunk with context
            chunks.append(
                DocAwareChunk(
                    source_document=document,
                    chunk_id=len(chunks),
                    blurb=self._extract_blurb(split_text),
                    content=chunk_text,
                    source_links={0: ""},
                    section_continuation=(i != 0),
                    title_prefix="",
                    metadata_suffix_semantic="",
                    metadata_suffix_keyword="",
                    mini_chunk_texts=self._get_mini_chunk_texts(split_text),
                )
            )

        return chunks

    def _extract_blurb(self, text: str) -> str:
        texts = self.blurb_splitter.split_text(text)
        if not texts:
            return ""
        return texts[0]

    def _get_mini_chunk_texts(self, chunk_text: str) -> list[str] | None:
        if self.mini_chunk_splitter and chunk_text.strip():
            return self.mini_chunk_splitter.split_text(chunk_text)
        return None

    def chunk(self, documents: list[Document]) -> list[DocAwareChunk]:
        """Process documents into semantically cohesive chunks using LLM"""
        final_chunks: list[DocAwareChunk] = []

        for document in documents:
            # Get document text and add line numbers
            document_text = "\n".join(section.text for section in document.sections)
            numbered_text = self._add_line_numbers(document_text)

            # Get metadata of the document
            metadata_suffix_keyword = ""
            if self.include_metadata:
                (
                    _,
                    metadata_suffix_keyword,
                ) = _get_metadata_suffix_for_document_index(document.metadata, include_separator=True)

            # Get document context
            document_context = self._get_document_context(document)

            # Get semantic sections using LLM
            sections = self.llm.analyze_sections(numbered_text)

            # Process each section
            for section in sections:
                section_text = self._extract_lines(document_text, section.start_line, section.end_line)
                section_context = self._get_section_context(section)

                # Split section into smaller chunks if needed
                tokens = self.tokenizer.tokenize(section_text)
                context = f"{document_context}\n\n{section_context}"
                context_tokens = self.tokenizer.tokenize(context)
                max_tokens_per_chunk = self.chunk_token_limit - len(context_tokens)
                if max_tokens_per_chunk <= CHUNK_MIN_CONTENT:
                    document_context = document_context[:256]
                    section_context = section_context[:256]
                    context = f"{document_context}\n\n{section_context}"
                    context_tokens = self.tokenizer.tokenize(context)
                    max_tokens_per_chunk = self.chunk_token_limit - len(context_tokens)

                chunk_text = f"{context}\n\n{section_text}"
                if len(self.tokenizer.tokenize(chunk_text)) <= max_tokens_per_chunk:
                    # Section fits in one chunk
                    print("SECTION LENGTH: ", len(self.tokenizer.tokenize(chunk_text)))
                    final_chunks.append(
                        DocAwareChunk(
                            source_document=document,
                            chunk_id=len(final_chunks),
                            blurb=self._extract_blurb(section_text),
                            content=chunk_text,
                            source_links={0: ""},
                            section_continuation=False,
                            title_prefix="",
                            metadata_suffix_semantic="",
                            metadata_suffix_keyword=metadata_suffix_keyword,
                            mini_chunk_texts=self._get_mini_chunk_texts(section_text),
                        )
                    )
                else:
                    # Split section into multiple chunks
                    start = 0
                    while start < len(tokens):
                        end = min(start + max_tokens_per_chunk - len(context_tokens), len(tokens))
                        chunk_text = " ".join(tokens[start:end])
                        full_chunk_text = f"{context}\n\n{chunk_text}"

                        print("CHUNK LENGTH: ", len(self.tokenizer.tokenize(full_chunk_text)))

                        final_chunks.append(
                            DocAwareChunk(
                                source_document=document,
                                chunk_id=len(final_chunks),
                                blurb=self._extract_blurb(chunk_text),
                                content=full_chunk_text,
                                source_links={0: ""},
                                section_continuation=(start > 0),
                                title_prefix="",
                                metadata_suffix_semantic="",
                                metadata_suffix_keyword=metadata_suffix_keyword,
                                mini_chunk_texts=self._get_mini_chunk_texts(chunk_text),
                            )
                        )
                        start = end

            if self.heartbeat:
                self.heartbeat.heartbeat()

        return final_chunks
