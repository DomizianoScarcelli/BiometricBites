import React, { useState, useEffect } from 'react';
import { ReactSession } from 'react-client-session';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import './GetFaces.scss'

function GetFaces() {
    const [userPhoto, setUserPhoto] = useState([])
    const navigate = useNavigate();

	useEffect (() => {
        ReactSession.setStoreType("sessionStorage");
		if (ReactSession.get("USER_EMAIL") === undefined)
		{
			navigate('/login');
		} else {
                axios.get('http://localhost:8000/api/get_photo_list', { params: { id: ReactSession.get('USER_ID') } })
                .then(function(response) {
                    setUserPhoto(JSON.parse(response.data.data));
                })

        }
	}, [])
	
	return (
		<>
            {userPhoto.map((item, index) => (
                <div className='main' key={index}>
                    <img src={'http://localhost:8000/samples/'+ReactSession.get('USER_ID')+'/'+item} alt={'user'+index}></img>
                </div>
            ))}
        </>
	)
}
//<img src={'http://localhost:8000/samples/'+ReactSession.get('USER_ID')+item} alt={'user'+{index}}></img>
export default GetFaces