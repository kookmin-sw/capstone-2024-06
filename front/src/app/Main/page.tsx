import Nav from "../components/Nav";
import DeskGL from "../components/DeskGL";

export default function Main() {
  return (
    <main className="flex-col w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
          <DeskGL />
        </div>
      </div>
    </main>
  );
}
