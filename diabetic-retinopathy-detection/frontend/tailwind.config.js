/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: { DEFAULT: '#008F87', dark: '#007A73', light: '#E0F7F5' },
        bg: '#EFF8FA',
        heading: '#172235',
        muted: '#64748B',
        surface: '#FFFFFF',
        border: '#E2E8F0',
      },
      fontFamily: {
        serif: ['Fraunces', 'serif'],
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
