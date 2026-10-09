import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Student Performance & Learning-Risk Analytics",
  description: "Academic insights, early-warning indicators, and model predictions."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
