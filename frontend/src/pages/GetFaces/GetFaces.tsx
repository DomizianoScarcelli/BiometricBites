import React, { useState, useEffect } from 'react';
import { ReactSession } from 'react-client-session';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import { BackButton, ProfileIconName } from '../../components';
import { ImBin } from 'react-icons/im';
import './GetFaces.scss'

function GetFaces() {
    const [ userPhoto, setUserPhoto ] = useState([])
    const navigate = useNavigate();

    const firstLetterUppercase = (str: string) => {
		return str.charAt(0).toUpperCase() + str.slice(1);
	}

    const deletePhoto = (id: number, photo: String) => {
        axios.delete('http://localhost:8000/api/delete_photo', { params: { id: id, name: photo } })
        .then(function(response) {
            refreshPhoto();
        })
    }

    const refreshPhoto = () => {
        axios.get('http://localhost:8000/api/get_photo_list', { params: { id: ReactSession.get('USER_ID') } })
        .then(function(response) {
            setUserPhoto(JSON.parse(response.data.data));
        })
    }

	useEffect (() => {
        ReactSession.setStoreType("sessionStorage");
		if (ReactSession.get("USER_EMAIL") === undefined)
		{
			navigate('/login');
		} else {
            refreshPhoto();
        }
	}, [])
	
	return (
        <div className='background'>
            <ProfileIconName name={ReactSession.get("USER_EMAIL") !== undefined ? firstLetterUppercase(ReactSession.get("USER_NAME"))+" "+firstLetterUppercase(ReactSession.get("USER_SURNAME")) : ''} />
            <BackButton link='/' />
            <div className='centralContainer'>
                <div className='photoContainer'>
                    <div className='photoContainerText'>
                        <p>Your Photos</p>
                    </div>
                    <div className='photoContainerItems'>
                        {userPhoto.map((item: String, index: number) => (
                            <div className='photoItem' key={index}>
                                <img src={'http://localhost:8000/samples/'+ReactSession.get('USER_ID')+'/'+item} alt={'user'+index}></img>
                                <button onClick={() => deletePhoto(ReactSession.get('USER_ID'), item)}><ImBin /></button>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
	)
}
export default GetFaces