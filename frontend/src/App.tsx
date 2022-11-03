import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import { Homepage, NoPage } from './pages';
import Layout from './pages/layout';
import './App.scss';

const App = () => {
  return (
    <div className="App">
      <BrowserRouter basename='/'>
        <Routes>
            <Route path='/' element={<Layout />}>
                <Route index element={<Homepage />} />
                <Route path='*' element={<NoPage />} />
            </Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
