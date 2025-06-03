import React from 'react';
import BoxList from './BoxList';
import './App.css';

function App() {
  return (
    <div className="App">
      <div className="app-container">
        <h1>Scrollable Box List with Pagination</h1>
        <BoxList />
      </div>
    </div>
  );
}

export default App;
