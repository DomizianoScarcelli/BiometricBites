import React from "react"

import "./Button.scss"

type ButtonProps = {
	text: string
	img: string
}

const Button: React.FC<ButtonProps> = (props) => {
	return (
		<>
			<button>
				<p>
					{/* Strange trick to use \n in the text props */}
					{props.text.split("\n").map((str) => (
						<p>{str}</p>
					))}
				</p>
				<img alt="face emoji" src={props.img}></img>
			</button>
		</>
	)
}

export default Button
