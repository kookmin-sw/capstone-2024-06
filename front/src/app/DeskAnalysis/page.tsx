import Nav from "../components/Nav";
import AnalysisImageUpLoader from "../components/AnalysisImageUpLoader";

export default function DeskAnalysis() {
  return (
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex  justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <AnalysisImageUpLoader />
        </div>
      </div>
    </main>
  );
}
