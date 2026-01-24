import jsPDF from "jspdf";
import html2canvas from "html2canvas";

type AnalysisResult = {
  riskLevel: string;
  confidence: number;
  sentiment: {
    positive: number;
    negative: number;
    neutral: number;
  };
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
    disgust: number;
  };
  recommendations: string[];
};

const AnalysisReport = ({ result }: { result: AnalysisResult }) => {
  const downloadPDF = async () => {
    const element = document.getElementById("report-content");
    if (!element) return;

    const canvas = await html2canvas(element, { scale: 2 });
    const imgData = canvas.toDataURL("image/png");

    const pdf = new jsPDF("p", "mm", "a4");
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

    pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
    pdf.save("analysis_report.pdf");
  };

  return (
    <div>
      {/* REPORT CONTENT */}
      <div
        id="report-content"
        style={{
          background: "#ffffff",
          color: "#111827",
          padding: "24px",
          borderRadius: "10px",
          border: "1px solid #e5e7eb",
          maxWidth: "800px",
          fontFamily: "Arial, sans-serif",
        }}
      >
        <h2 style={{ color: "#2563eb" }}>
          Mental Health Analysis Report
        </h2>

        <hr style={{ margin: "12px 0" }} />

        {/* Summary */}
        <p>
          <strong>Risk Level:</strong>{" "}
          <span
            style={{
              color: result.riskLevel === "high" ? "#dc2626" : "#16a34a",
              fontWeight: "bold",
            }}
          >
            {result.riskLevel}
          </span>
        </p>

        <p>
          <strong>Confidence:</strong>{" "}
          <span style={{ color: "#0f766e" }}>
            {result.confidence.toFixed(2)}%
          </span>
        </p>

        {/* Sentiment */}
        <div
          style={{
            background: "#f1f5f9",
            padding: "12px",
            borderRadius: "8px",
            marginTop: "16px",
          }}
        >
          <h3 style={{ color: "#1e40af" }}>Sentiment</h3>
          <ul>
            <li>Positive: {result.sentiment.positive}</li>
            <li>Negative: {result.sentiment.negative}</li>
            <li>Neutral: {result.sentiment.neutral}</li>
          </ul>
        </div>

        {/* Emotions */}
        <div
          style={{
            background: "#fef3c7",
            padding: "12px",
            borderRadius: "8px",
            marginTop: "16px",
          }}
        >
          <h3 style={{ color: "#92400e" }}>Emotions</h3>
          <ul>
            <li>Joy: {result.emotions.joy}</li>
            <li>Sadness: {result.emotions.sadness}</li>
            <li>Anger: {result.emotions.anger}</li>
            <li>Fear: {result.emotions.fear}</li>
            <li>Surprise: {result.emotions.surprise}</li>
            <li>Disgust: {result.emotions.disgust}</li>
          </ul>
        </div>

        {/* Recommendations */}
        <div
          style={{
            background: "#ecfeff",
            padding: "12px",
            borderRadius: "8px",
            marginTop: "16px",
          }}
        >
          <h3 style={{ color: "#0e7490" }}>Recommendations</h3>
          <ul>
            {result.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>

        <p
          style={{
            fontSize: "12px",
            marginTop: "20px",
            color: "#6b7280",
          }}
        >
          *This report is generated for monitoring purposes only.
        </p>
      </div>

      {/* PDF BUTTON */}
      <button
        onClick={downloadPDF}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          background: "#2563eb",
          color: "#ffffff",
          borderRadius: "6px",
          cursor: "pointer",
        }}
      >
        Download PDF
      </button>
    </div>
  );
};

export default AnalysisReport;
