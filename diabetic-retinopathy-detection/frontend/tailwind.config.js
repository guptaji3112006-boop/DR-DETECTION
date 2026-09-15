/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: { DEFAULT: '#0077B6', dark: '#03045E', light: '#CAF0F8' },
        bg: '#F8FAFC',
        heading: '#03045E',
        muted: '#475569',
        surface: '#FFFFFF',
        border: '#90E0EF',
      },
      fontFamily: {
        serif: ['Fraunces', 'serif'],
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
