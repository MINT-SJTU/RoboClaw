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
        sf2: 'var(--sf2)',
        bd: 'var(--bd)',
        bd2: 'var(--bd2)',
        tx: 'var(--tx)',
        tx2: 'var(--tx2)',
        tx3: 'var(--tx3)',
        ac: 'var(--ac)',
        ac2: 'var(--ac2)',
        gn: 'var(--gn)',
        rd: 'var(--rd)',
        yl: 'var(--yl)',
        or: 'var(--or)',
      },
      fontFamily: {
        base: ['Outfit', '-apple-system', 'sans-serif'],
        mono: ["'JetBrains Mono'", "'SF Mono'", 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['11px', { lineHeight: '1.4' }],
        'xs': ['12px', { lineHeight: '1.5' }],
        'sm': ['13px', { lineHeight: '1.5' }],
        'base': ['14px', { lineHeight: '1.6' }],
        'lg': ['16px', { lineHeight: '1.5' }],
        'xl': ['20px', { lineHeight: '1.3', letterSpacing: '-0.01em' }],
        '2xl': ['24px', { lineHeight: '1.2', letterSpacing: '-0.02em' }],
      },
      borderRadius: {
        sm: '4px',
        DEFAULT: '8px',
        lg: '12px',
        xl: '16px',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'elevated': '0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06)',
        'glow-ac': '0 0 0 3px rgba(37,99,235,0.1)',
      },
    },
  },
  plugins: [],
}
