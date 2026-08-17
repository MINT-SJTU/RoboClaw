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
        base: ['Manrope', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
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
      boxShadow: {
        card: '0 18px 34px rgba(47, 111, 228, 0.09)',
        'card-hover': '0 24px 42px rgba(47, 111, 228, 0.14)',
        elevated: '0 22px 44px rgba(47, 111, 228, 0.14)',
        'glow-ac': '0 16px 30px rgba(47, 111, 228, 0.22)',
        'glow-gn': '0 16px 30px rgba(47, 111, 228, 0.18)',
        'glow-rd': '0 16px 30px rgba(47, 111, 228, 0.18)',
        'glow-yl': '0 16px 30px rgba(47, 111, 228, 0.18)',
        'inset-ac': 'inset 0 0 0 1px rgba(47, 111, 228, 0.14), 0 18px 34px rgba(47, 111, 228, 0.08)',
        'inset-gn': 'inset 0 0 0 1px rgba(47, 111, 228, 0.14), 0 18px 34px rgba(47, 111, 228, 0.08)',
        'inset-yl': 'inset 0 0 0 1px rgba(47, 111, 228, 0.14), 0 18px 34px rgba(47, 111, 228, 0.08)',
      },
    },
  },
  plugins: [],
}
