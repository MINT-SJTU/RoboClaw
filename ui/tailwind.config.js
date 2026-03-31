/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        sf: 'var(--sf)',
        bd: 'var(--bd)',
        tx: 'var(--tx)',
        tx2: 'var(--tx2)',
        ac: 'var(--ac)',
        gn: 'var(--gn)',
        rd: 'var(--rd)',
        yl: 'var(--yl)',
        or: 'var(--or)',
      },
      fontFamily: {
        base: ['-apple-system', "'Segoe UI'", 'Roboto', 'sans-serif'],
        mono: ["'SF Mono'", 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': '11px',
        'xs': '12px',
        'sm': '13px',
        'base': '15px',
        'lg': '17px',
      },
      borderRadius: {
        sm: '4px',
        DEFAULT: '6px',
        lg: '8px',
      },
    },
  },
  plugins: [],
}
