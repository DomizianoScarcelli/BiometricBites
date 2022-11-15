import React from "react"

import "./Button.scss"

type ButtonProps = {
	text: string
	img: string
	shadow: boolean
}

const Button: React.FC<ButtonProps> = (props) => {
	return (
		<>
			<button style={props.shadow ? { boxShadow: "4px 5px 19px rgba(var(--shadow-color-rgb), 0.5)" } : {}}>
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
