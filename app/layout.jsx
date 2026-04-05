import './globals.css'

export const metadata = {
  title: 'Training Lab',
  description: 'Ideas, experiments, benchmark notes, and daily TILs from Training Lab.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  )
}
