import Nav from "../components/Nav";


export default function MyPage() {
  return (
    <main className="flex-col justify-center border-b w-full h-full">
      <Nav />
      <div className="flex justify-center w-full h-auto">
        <div className="flex-col items-center min-w-[700px] max-w-[1000px] w-11/12 h-auto">
        </div>
      </div>
    </main>
  );
}
