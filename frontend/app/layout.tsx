import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PredictEPL - AI-Powered EPL Predictions",
  description:
    "Machine learning powered English Premier League score predictions for Nigerian football fans",
};

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      className="text-sm text-slate-400 hover:text-white transition-colors px-3 py-2 rounded-lg hover:bg-white/5"
    >
      {children}
    </a>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen" suppressHydrationWarning>
        <nav className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
            <a href="/" className="flex items-center gap-2">
              <span className="text-lg font-bold text-white">
                Predict<span className="text-green-500">EPL</span>
              </span>
            </a>
            <div className="flex items-center gap-1">
              <NavLink href="/">Matches</NavLink>
              <NavLink href="/accuracy">Accuracy</NavLink>
              <NavLink href="/standings">Table</NavLink>
            </div>
          </div>
        </nav>
        <main className="max-w-5xl mx-auto px-4 py-6">{children}</main>
        <footer className="border-t border-slate-800 mt-12 py-6 text-center text-sm text-slate-500">
          PredictEPL &mdash; AI-powered predictions. Not financial advice.
        </footer>
      </body>
    </html>
  );
}
