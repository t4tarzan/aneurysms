# Aneurysm Detection and 3D Visualization

This project provides a web-based 3D visualization tool for aneurysm detection, featuring a Next.js frontend with Three.js for 3D rendering.

## Features

- 3D brain visualization with interactive controls
- Aneurysm detection visualization
- Support for different types of aneurysms (saccular, fusiform, mycotic)
- Responsive design that works on desktop and mobile
- TypeScript support for better code quality

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn
- Python (for backend services, if applicable)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   cd web-app
   npm install  # or yarn install
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
/
├── web-app/                 # Next.js frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Next.js pages
│   │   └── styles/         # Global styles
│   ├── public/             # Static files
│   └── package.json        # Frontend dependencies
├── generate_code/          # Python code for post-processing
│   └── aneurysm_detection_postprocessing/
│       └── postprocessing/ # Post-processing utilities
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Development

### Available Scripts

- `npm run dev` - Start the development server
- `npm run build` - Build for production
- `npm start` - Start the production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
