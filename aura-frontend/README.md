# AURA Federated Learning Security Dashboard

A comprehensive React TypeScript frontend for the AURA (Auditable, Unified, Resilient Architecture) Federated Learning Security System. This dashboard provides real-time monitoring, analysis, and visualization of federated learning model submissions with integrated XAI-based security analysis and blockchain verification.

## Features

### 🎯 Core Functionality

- **Dashboard**: Real-time overview with statistics, charts, and recent activity monitoring
- **Model Upload**: Drag-and-drop interface for submitting federated learning models with multi-step progress tracking
- **SHAP Reports**: Detailed behavioral analysis reports with feature importance visualizations
- **Blockchain Ledger**: Immutable transaction history with filtering and verification capabilities
- **Analytics**: Time-based metrics, performance trends, and hospital rankings
- **Security Center**: Real-time threat monitoring, security recommendations, and system health tracking

### 🎨 Design & UX

- **Professional Medical/Tech Aesthetic**: Clean, modern interface designed for healthcare security professionals
- **Dark Mode Support**: Seamless light/dark theme switching with system preference detection
- **Responsive Design**: Fully responsive layout optimized for desktop, tablet, and mobile devices
- **Smooth Animations**: Framer Motion animations for page transitions and component interactions
- **Accessible**: ARIA labels, keyboard navigation, and screen reader support

### 🛠️ Technical Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Routing**: React Router v6
- **Styling**: Tailwind CSS v3 with custom theme
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **API Client**: Axios (with mock implementation)

## Project Structure

```
aura-frontend/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── Alert.tsx
│   │   ├── ChartContainer.tsx
│   │   ├── Layout.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── Navbar.tsx
│   │   ├── Sidebar.tsx
│   │   └── StatCard.tsx
│   ├── pages/            # Page components
│   │   ├── Dashboard.tsx
│   │   ├── Upload.tsx
│   │   ├── Reports.tsx
│   │   ├── Ledger.tsx
│   │   ├── Analytics.tsx
│   │   └── Security.tsx
│   ├── services/         # API and mock data services
│   │   ├── api.ts
│   │   └── mockData.ts
│   ├── hooks/            # Custom React hooks
│   │   ├── useApi.ts
│   │   ├── useTheme.ts
│   │   └── useWebSocket.ts
│   ├── utils/            # Utility functions
│   │   ├── cn.ts
│   │   ├── format.ts
│   │   └── validation.ts
│   ├── types/            # TypeScript type definitions
│   │   └── index.ts
│   ├── App.tsx           # Main app component with routing
│   ├── main.tsx          # Application entry point
│   └── index.css         # Global styles with Tailwind directives
├── public/               # Static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

## Getting Started

### Prerequisites

- Node.js 20.19+ or 22.12+
- npm, yarn, or pnpm

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd aura-frontend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Color Palette

The application uses a carefully selected color palette for a professional medical/tech aesthetic:

- **Primary**: Indigo (#4f46e5) - Used for primary actions and branding
- **Success**: Emerald (#10b981) - Approved models and positive states
- **Warning**: Amber (#f59e0b) - Processing states and warnings
- **Danger**: Rose (#f43f5e) - Rejected models and critical alerts

## Key Components

### Dashboard
- Real-time statistics cards with trend indicators
- Line chart for recent activity (7 days)
- Pie chart for security metrics distribution
- Recent submissions table

### Upload
- Drag-and-drop file upload with validation
- Multi-step progress indicator (Upload → Analysis → Detection → Ledger)
- Form fields for hospital ID, description, and privacy level
- Result display with submission details and action buttons

### Reports
- Grid of behavioral report cards
- Detailed report view with SHAP analysis
- Behavioral fingerprint bar chart visualization
- Security assessment metrics

### Ledger
- Transaction filtering by hospital and verdict
- Paginated transaction table
- Blockchain verification functionality
- Blockchain information panel

### Analytics
- Time range selector (7d, 30d, 90d)
- Multiple chart types (line, bar, area)
- Top performing hospitals ranking
- Attack type distribution

### Security
- Threat status filtering
- Real-time threat detection timeline
- Security recommendations with priority levels
- System health monitoring dashboard

## Mock Data

The application includes comprehensive mock data generation for demonstration purposes:

- Realistic behavioral reports with SHAP fingerprints
- Transaction history with various verdicts
- Daily metrics and trends
- Attack type distributions
- Hospital performance data
- Security threats and recommendations

## API Integration

The current implementation uses mock APIs with simulated delays and error handling. To integrate with the real AURA backend:

1. Update the `BASE_URL` in `src/services/api.ts`
2. Replace mock implementations with actual API calls
3. Update type definitions if needed

## Customization

### Theme

Edit `tailwind.config.js` to customize colors, fonts, and other design tokens.

### Mock Data

Modify `src/services/mockData.ts` to adjust mock data generation patterns.

### API Endpoints

Update `src/services/api.ts` to change API endpoints or add new ones.

## Performance

- Code splitting for optimal bundle size
- Lazy loading for route-based code splitting
- Optimized re-renders with React best practices
- Efficient chart rendering with Recharts

## Accessibility

- Semantic HTML structure
- ARIA labels on interactive elements
- Keyboard navigation support
- Screen reader compatible
- Focus management

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is part of the AURA Federated Learning Security System.

## Contributing

1. Follow the existing code structure and naming conventions
2. Ensure TypeScript types are properly defined
3. Add comments for complex logic
4. Test across different screen sizes and themes
5. Maintain accessibility standards
