from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import os


class DocumentIntelligenceImpl:
    def __init__(
        self,
        documentintelligence_endpoint: str,
        documentintelligence_api_key: str,
        filename: str,
    ):
        self.endpoint = documentintelligence_endpoint
        self.key = documentintelligence_api_key
        self.filename = filename
        self.document_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )

    # @staticmethod
    def _in_span(self, word, spans):
        for span in spans:
            if word.span.offset >= span.offset and (
                word.span.offset + word.span.length
            ) <= (span.offset + span.length):
                return True
        return False

    # @staticmethod
    def _format_polygon(self, polygon):
        if not polygon:
            return "N/A"
        return ", ".join(
            [f"[{polygon[i]}, {polygon[i + 1]}]" for i in range(0, len(polygon), 2)]
        )

    def process_document(self):
        with open(self.filename, "rb") as f:
            poller = self.document_intelligence_client.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=f,
                content_type="application/octet-stream",
            )
            result: AnalyzeResult = poller.result()
            with open(str.replace(self.filename, ".pdf", ".txt"), "w") as fw:
                if result.styles and any(
                    [style.is_handwritten for style in result.styles]
                ):
                    print("Detected handwritten content")
                    # fw.write("Document contains handwritten content")
                else:
                    print("Document does not contain handwritten content")
                    # fw.write("Document does not contain handwritten content")

                for page in result.pages:
                    print(f"----Analyzing layout from page #{page.page_number}----")
                    fw.write(f"page #{page.page_number}----")
                    print(
                        f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
                    )
                    # fw.write(
                    #     f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
                    # )

                    if page.lines:
                        for line_idx, line in enumerate(page.lines):
                            words = []
                            if page.words:
                                for word in page.words:
                                    print(
                                        f"......Word '{word.content}' has a confidence of {word.confidence}"
                                    )
                                    # fw.write(
                                    #     f"......Word '{word.content}' has a confidence of {word.confidence}"
                                    # )
                                    if self._in_span(word, line.spans):
                                        words.append(word)
                            print(
                                f"...Line # {line_idx} has word count {len(words)} and text '{line.content}' "
                                f"within bounding polygon '{self._format_polygon(line.polygon)}'"
                            )
                            fw.write(
                                f"{line.content}"
                                # f"within polygon '{self._format_polygon(line.polygon)}'"
                            )

                    if page.selection_marks:
                        for selection_mark in page.selection_marks:
                            print(
                                f"Selection mark is '{selection_mark.state}' within bounding polygon "
                                f"'{self._format_polygon(selection_mark.polygon)}' and has a confidence of {selection_mark.confidence}"
                            )
                            # fw.write(
                            #     f"Selection mark is '{selection_mark.state}' within bounding polygon "
                            #     f"'{self._format_polygon(selection_mark.polygon)}' and has a confidence of {selection_mark.confidence}"
                            # )

                if result.paragraphs:
                    print(
                        f"----Detected #{len(result.paragraphs)} paragraphs in the document----"
                    )
                    fw.write(
                        f"----Detected #{len(result.paragraphs)} paragraphs in the document----"
                    )
                    # Sort all paragraphs by span's offset to read in the right order.
                    result.paragraphs.sort(
                        key=lambda p: (
                            p.spans.sort(key=lambda s: s.offset),
                            p.spans[0].offset,
                        )
                    )
                    print("-----Print sorted paragraphs-----")
                    # fw.write("-----Print sorted paragraphs-----")
                    for paragraph in result.paragraphs:
                        if not paragraph.bounding_regions:
                            print(
                                f"Found paragraph with role: '{paragraph.role}' within N/A bounding region"
                            )
                            # fw.write(
                            #     f"Found paragraph with role: '{paragraph.role}' within N/A bounding region"
                            # )
                        else:
                            print(
                                f"Found paragraph with role: '{paragraph.role}' within"
                            )
                            # fw.write(
                            #     f"Found paragraph with role: '{paragraph.role}' within"
                            # )
                            print(
                                ", ".join(
                                    f" Page #{region.page_number}: {self._format_polygon(region.polygon)} bounding region"
                                    for region in paragraph.bounding_regions
                                )
                            )
                            # fw.write(
                            #     ", ".join(
                            #         f" Page #{region.page_number}: {self._format_polygon(region.polygon)} bounding region"
                            #         for region in paragraph.bounding_regions
                            #     )
                            # )
                        print(f"...with content: '{paragraph.content}'")
                        fw.write(f"...with content: '{paragraph.content}'")
                        print(
                            f"...with offset: {paragraph.spans[0].offset} and length: {paragraph.spans[0].length}"
                        )
                        # fw.write(
                        #     f"...with offset: {paragraph.spans[0].offset} and length: {paragraph.spans[0].length}"
                        # )

                if result.tables:
                    for table_idx, table in enumerate(result.tables):
                        print(
                            f"Table # {table_idx} has {table.row_count} rows and "
                            f"{table.column_count} columns"
                        )
                        fw.write(
                            f"DETECTING Table # {table_idx} has {table.row_count} rows and "
                            f"{table.column_count} columns"
                        )
                        if table.bounding_regions:
                            for region in table.bounding_regions:
                                print(
                                    f"Table # {table_idx} location on page: {region.page_number} is {self._format_polygon(region.polygon)}"
                                )
                                # fw.write(
                                #     f"Table # {table_idx} location on page: {region.page_number} is {self._format_polygon(region.polygon)}"
                                # )
                        for cell in table.cells:
                            print(
                                f"...Cell[{cell.row_index}][{cell.column_index}] has text '{cell.content}'"
                            )
                            fw.write(f"'{cell.content}'")
                            if cell.bounding_regions:
                                for region in cell.bounding_regions:
                                    print(
                                        f"...content on page {region.page_number} is within bounding polygon '{self._format_polygon(region.polygon)}'"
                                    )
                                    # fw.write(
                                    #     f"...content on page {region.page_number} is within bounding polygon '{self._format_polygon(region.polygon)}'"
                                    # )

        print("----------------------------------------")


if __name__ == "__main__":
    dii = DocumentIntelligenceImpl(
        "https://documentintelligence.cognitiveservices.azure.com/",
        "BZviyUon03opew2djzG1ruTUeJG4d908OJJKjYUZTFi3MgOt4zDdJQQJ99ALACYeBjFXJ3w3AAALACOGWABT",
        r"C:\Users\bganu\OneDrive\Desktop\Data_Science_Naresh_Technologies\BB_AI_PROJECTS\bbprojects\copilotbot\app\data\BoardingPass_AirIndia.pdf",
    )
    dii.process_document()
