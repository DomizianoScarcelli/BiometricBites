import React, { useState, useEffect } from 'react';
import { ReactSession } from 'react-client-session';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import './GetFaces.scss'

function GetFaces() {
    const { state } = useLocation();
    const { userPhoto } = state
    const navigate = useNavigate();

	useEffect (() => {
        ReactSession.setStoreType("sessionStorage");
		if (ReactSession.get("USER_EMAIL") === undefined)
		{
			navigate('/login');
		}
	}, [])
	
	return (
		<>
            {userPhoto.map((item: String, index: number) => (
                <div className='main' key={index}>
                    <img src={'http://localhost:8000/samples/'+ReactSession.get('USER_ID')+'/'+item} alt={'user'+index}></img>
                </div>
            ))}
        </>
	)
}
//<img src={'http://localhost:8000/samples/'+ReactSession.get('USER_ID')+item} alt={'user'+{index}}></img>
export default GetFaces