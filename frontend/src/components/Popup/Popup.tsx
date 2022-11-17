import React from 'react';

import './Popup.scss';

const Popup = (props: any) => {
  return (props.trigger) ? (
    <>
      <div className='popup-shade'>
        <div className='popup-inner'>
          <div className='popup-title'>
              {props.title}
          </div>
          <div className='popup-description'>
            {props.description}
          </div>
          {props.children}
          <div className='popup-button'>
            <button type='button' onClick={props.onClick}>Close</button>
          </div>
        </div>
      </div>
    </>
  ) : (<></>);
}

export default Popup
