import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ReactSession } from 'react-client-session';

import { Homepage, LoginPage, NoPage } from './pages';
import Layout from './pages/layout';
import './App.scss';

const App = () => {
  ReactSession.setStoreType('sessionStorage');

  return (
    <div className="App">
      <BrowserRouter basename='/'>
        <Routes>
            <Route path='/' element={<Layout />}>
                <Route index element={<Homepage />} />
                <Route path='login' element={<LoginPage />} />
                <Route path='*' element={<NoPage />} />
            </Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
